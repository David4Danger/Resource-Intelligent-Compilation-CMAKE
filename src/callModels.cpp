/*
 * CUDA for Prognostics - Models & Model call wrapper functions
 *
 * Implements prognostic model functions and their respective wrappers
 * to support architecture-based compilation.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/

#include "callModels.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>

#ifdef __CUDACC__
#include "debugCFP.h"
#include "prng.h"
#else
#include <random>
#endif // Compilation specific libraries


// Define qualifiers, also confirms which compiler is being used.
#ifdef __CUDACC__
  #warning using nvcc
  #define HOSTQUALIFIER __host__
  #define DEVICEQUALIFIER __device__
  #define HOSTDEVICEQUALIFIER __host__ __device__
  #define KERNEL __global__
#else
  #warning using g++
  #define HOSTQUALIFIER
  #define DEVICEQUALIFIER
  #define HOSTDEVICEQUALIFIER
  #define KERNEL
#endif

/* Runs a monte carlo simulation of stock prices over time. Prices
 * all start at a specific fixed value and then update on a daily
 * basis until they hit a certain price (upper and lower bounds)
 * or the time horizon.
*/
HOSTDEVICEQUALIFIER
static void stockModel(int particles,
                       int timeStep,
                       int timeHorizon,
                       double stockPrice,
                       double stockMin,
                       double stockMax,
                       double *stdv,
                       double *exreturn,
                       double *shockBools,
                       int *dayHisto) {
  // Declare vars
  double drift, shockVal, curPrice, deltaPrice;
#ifdef __CUDA_ARCH__ // Decide on iteration technique
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < particles; i += stride) {
#else
  for (int i = 0; i < particles; i++) {
#endif //iteration decision
    curPrice = stockPrice;
    drift = exreturn[i] * 0.1 * (double)timeStep;
    shockVal = stdv[i] * 0.1 * sqrt((double)timeStep);
    int t;

    for (t = 0; t < timeHorizon; t+= timeStep) {
      // move forward a step
      if (shockBools[t * i] < 0.50) {
        deltaPrice = curPrice * (drift - shockVal);
      } else {
        deltaPrice = curPrice * (drift + shockVal);
      }
      curPrice += deltaPrice;

      // Check if stock price exceeds a boundary
      bool boundaryPassed = (curPrice > stockMax) || (curPrice < stockMin);
      if (boundaryPassed) {
#ifdef __CUDA_ARCH__ // Avoid race conditions when writing to histogram
        atomicAdd(&dayHisto[t], 1);
#else
        dayHisto[t] += 1;
#endif
        break;
      }
    }

    if (t >= timeHorizon) {
      // Never broke the boundary, just add to last day
#ifdef __CUDA_ARCH__ // Again avoid race conditions
      atomicAdd(&dayHisto[timeHorizon - timeStep], 1);
#else
      dayHisto[timeHorizon - timeStep] += 1;
#endif
    }
  }
}

/* GPU wrapper needs this extra layer as it needs to invoke a __global__
 * function to call the model which is qualified by host and device.
 * This exists on a per-thread basis. Just pass things on to the model.
*/
KERNEL //quantifier indicates device entry point
static void stockModelCallKern (int particles, int timeStep, int timeHorizon,
                         double stockPrice, double stockMin, double stockMax,
                         double *stdv, double *exreturn, double *shockBools,
                         int *dayHisto) {

  stockModel(particles, timeStep, timeHorizon, stockPrice, stockMin,
             stockMax, stdv, exreturn, shockBools, dayHisto);
}

// CPU wrapper function for the stock model. Only invoked on CPU arch.
static void stockModelCPU (int particles, int timeStep, int timeHorizon,
                    double stockPrice, double stockMin, double stockMax,
                    double *stdv, double *exreturn, double *shockBools,
                    int *dayHisto) {

  stockModel(particles, timeStep, timeHorizon, stockPrice, stockMin,
             stockMax, stdv, exreturn, shockBools, dayHisto);
}

#ifdef __CUDACC__
// GPU wrapper function for the stock model. Only invoked on GPU arch.
static void stockModelGPU (int particles, int timeStep, int timeHorizon,
                    double stockPrice, double stockMin, double stockMax,
                    double *stdv, double *exreturn, double *shockBools,
                    int *dayHisto, int blocksPerGrid, int threadsPerBlock) {

  stockModelCallKern<<<blocksPerGrid, threadsPerBlock>>>
                    (particles, timeStep, timeHorizon, stockPrice, stockMin,
                     stockMax, stdv, exreturn, shockBools, dayHisto);
}
#endif


/* This function makes a call to the stockModel function, which can either be
 * for CPU-only achitecture or alternatively the accelerated GPU version using
 * CUDA.
 *
 * Should be used as a sample/demo of how to modify existing models to support
 * dynamic compilation depending on the system used and it's available hardware.
*/
void callStockModel(int blocksPerGrid, int threadsPerBlock,
                    int particles, int timeStep, int timeHorizon) {
  double stockPrice = 20.00;
  double stockMin = 3.75;
  double stockMax = 324.25;
  double *stdv;
  double *exreturn;
  double *shock;
  int *dayHisto;

#ifdef __CUDACC__ // Allocate memory (specific to device or host)
  DEBUG(DB_CUDAMEM, "Attempting to allocate memory for stock simulation, and zeroing out memory for histogram");
  handleCudaErr(cudaMalloc((void **)&stdv, sizeof(double) * particles));
  handleCudaErr(cudaMalloc((void **)&exreturn, sizeof(double) * particles));
  handleCudaErr(cudaMalloc((void **)&shock, sizeof(double) * particles * timeHorizon));
  handleCudaErr(cudaMalloc((void **)&dayHisto, sizeof(int) * timeHorizon));
  handleCudaErr(cudaMemset((void *)dayHisto, 0, sizeof(int) * timeHorizon));
  DEBUG(DB_CUDAMEM, "Succesfully allocated memory for stock simulation.");
#else
  stdv = (double *)malloc(sizeof(double) * particles);
  exreturn = (double *)malloc(sizeof(double) * particles);
  shock = (double *)malloc(sizeof(double) * particles * timeHorizon);
  dayHisto = (int *)calloc(timeHorizon, sizeof(int));//memset to 0
#endif // alloc memory

#ifdef __CUDACC__ // Generate random numbers
  prngUniformDouble(stdv, particles);
  prngUniformDouble(exreturn, particles);
  prngUniformDouble(shock, particles * timeHorizon);
  DEBUG(DB_CURAND, "Generated random doubles for stock simulation in parallel.");
#else
  std::random_device prng;
  std::mt19937 eng(prng()); //Mersenne Twister
  std::uniform_real_distribution<> gen(0,1);
  for (int i = 0; i < particles; ++i) {
    stdv[i] = gen(eng);
    exreturn[i] = gen(eng);
  }
  for (int j = 0; j < particles * timeHorizon; ++j) {
    shock[j] = gen(eng);
  }
#endif // rng

  // Begin timer
  std::clock_t start;
  double duration;
  start = std::clock();

// Call the proper function based on architecture. GPU has extra params
#ifdef __CUDACC__
  (void)stockModelCPU;
  std::cout << "Calling GPU version of model!" << std::endl;
  stockModelGPU(particles, timeStep, timeHorizon, stockPrice, stockMin,
                stockMax, stdv, exreturn, shock, dayHisto,
                blocksPerGrid, threadsPerBlock);
#else
  (void)blocksPerGrid;
  (void)threadsPerBlock;
  std::cout << "Calling CPU version of model!" << std::endl;
  stockModelCPU(particles, timeStep, timeHorizon, stockPrice, stockMin,
                stockMax, stdv, exreturn, shock, dayHisto);
#endif

  // End the timer
  duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  int daySum = 0;

#ifdef __CUDACC__ // Synch and copy back to host, both modes output
  DEBUG(DB_SYNC, "Waiting for all CUDA threads to finish execution...");
  cudaDeviceSynchronize();
  DEBUG(DB_SYNC, "...Done");
  int *h_dayHisto = (int *)malloc(sizeof(int) * timeHorizon);
  handleCudaErr(cudaMemcpy(h_dayHisto, dayHisto, sizeof(int) * timeHorizon, cudaMemcpyDeviceToHost));
  for (int i = 0; i < timeHorizon; ++i) {
    daySum += h_dayHisto[i] * i;
  }
#else // No copying back is necessary on the CPU
  for (int i = 0; i < timeHorizon; ++i) {
    daySum += dayHisto[i] * i;
  }
#endif

  std::cout << "Mean day boundary hit: " << daySum / particles << std::endl;
  std::cout << "Duration: " << duration << std::endl;

#ifdef __CUDACC__ // Free up memory
  cudaFree(stdv);
  cudaFree(exreturn);
  cudaFree(shock);
  cudaFree(dayHisto);
  DEBUG(DB_CUDAMEM, "Succesfully freed all allocated device memory.");
  free(h_dayHisto);
#else
  free(stdv);
  free(exreturn);
  free(shock);
  free(dayHisto);
#endif
}
