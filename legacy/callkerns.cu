/*
 * CUDA for Prognostics - Kernel call wrapper functions
 *
 * Implements all wrapper functions which call device kernel functions,
 * defined in kernels.cu.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/

#include "kernels.h"
#include "callkerns.h"
#include "../support/debugCFP.h"
#include "../support/prng.h"
#include <ctime>
#include <iostream>

/*
 * IMPORTANT NOTICE!!
 * The proper CUDA enabled device should be set in this module. For the
 * case of the development Jetson only one device is available, but more
 * advanced systems could make use of multiple available CUDA enabled
 * devices. Of course, using multiple devices while more difficult to
 * program for is absolutely an option.
*/

cudaDeviceProp deviceProp;
// For later - investigate time cost overhead of getting device info and
// verifying that threadsPerBlock is a multiple of device warp size. Could
// probably be predetermined and embedded into program for certain devices
// i.e: this code is going on device X, which has a warp size of 32.
//handleCudaErr(cudaGetDeviceProperties(&deviceProp, 0));

/* Use this as a guide when writing new wrappers. Adjust as necessary.
 * Begin by removing the block comments for each section and removing the unecessary
 * ones. Ideally most of the template is used and only additions are made.
*/
void genericKernelWrapper (int threadsPerBlock,
                           int blocksPerGrid,
                           int particles,
                           int timeStep,
                           int timeHorizon) {
  // Declare variables

  // Allocate and populate input arrays for device, as well as output.
  /*handleCudaErr(cudaMalloc((void **)&d_inputArr, sizeof(inputType) * numInputs));
  handleCudaErr(cudaMalloc((void **)&d_resultsArr, sizeof(outputType) * numOutputs));
  handleCudaErr(cudaMemset((void *)d_resultsArr, -1, sizeof(outputType) * numOutputs));
  DEBUG(DB_CUDAMEM, "Allocated device memory for generic kernel");*/


  // Begin timer
  /*std::clock_t start;
  double duration;
  start = std::clock();*/

  // Copy input to the device (if necessary)
  /*handleCudaErr(cudaMemcpy(d_location, h_location, sizeof(inputType) * numInputs, cudaMemcpyHostToDevice));
  DEBUG(DB_CUDAMEM, "Moved input over to device memory for generic kernel");*/

  // Make a call to a kernel in kernels.cu
  /*genericKernelCall<<<blocksPerGrid, threadsPerBlock>>>(particles, timeStep, timeHorizon);*/

  // Ensure all threads hit completion
  /*cudaDeviceSynchronize();*/

  // Copy results back from GPU.
  /*handleCudaErr(cudaMemcpy(h_location, d_location, sizeof(inputType) * numInputs, cudaMemcpyDeviceToHost));
  DEBUG(DB_CUDAMEM, "Moved kernel results back from device to host");*/

  // Manipulate results as necessary, stop timer.
  // Reduce only if not performed on GPU.
  /*duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;*/

  // Output Results

  // Free device and host memory
  /*cudaFree(DEVICE THINGS);
  free(HOST THINGS);
  DEBUG(DB_CUDAMEM, "Successfully freed all device and host memory");*/
}

////////////////////////////////////////////////////////////////////////////////

/* Kernel wrapper function for a basic monte carlo simulation for calculating stock
 * price. Refer to header for more details.
*/
void runStockSimKernel (int threadsPerBlock,
                        int blocksPerGrid,
                        int particles,
                        int timeStep,
                        int timeHorizon) {
  // Declare variables. Note that standard deviation and expected return are
  // randomly generated below.
  double stockPrice = 20.00;
  double stockMin = 3.75;
  double stockMax = 324.25;
  double *d_stdv;
  double *d_exreturn;
  // Next variable used to determine if shock is + or -; need particles * timeH
  double *d_shock;
  int *d_dayHisto;

  // Allocate and populate input arrays for device and output histogram.
  DEBUG(DB_CUDAMEM, "Attempting to allocate memory on device for standard deviation, expected return, shocks, and finish time histogram.");
  handleCudaErr(cudaMalloc((void **)&d_stdv, sizeof(double) * particles));
  handleCudaErr(cudaMalloc((void **)&d_exreturn, sizeof(double) * particles));
  handleCudaErr(cudaMalloc((void **)&d_shock, sizeof(double) * particles * timeHorizon));
  handleCudaErr(cudaMalloc((void **)&d_dayHisto, sizeof(double) * timeHorizon));
  handleCudaErr(cudaMemset((void *)d_dayHisto, 0, sizeof(int) * timeHorizon));
  DEBUG(DB_CUDAMEM, "Allocated device memory for Stock kernel without issue");

  prngUniformDouble(d_stdv, particles);
  prngUniformDouble(d_exreturn, particles);
  prngUniformDouble(d_shock, particles * timeHorizon);

  // Begin timer
  std::clock_t start;
  double duration;
  start = std::clock();

  // Call the stock kernel with variables from the command line.
  stockSimKernel<<<blocksPerGrid, threadsPerBlock>>>(particles,
                                                     timeStep,
                                                     timeHorizon,
                                                     stockPrice,
                                                     stockMin,
                                                     stockMax,
                                                     d_stdv,
                                                     d_exreturn,
                                                     d_shock,
                                                     d_dayHisto);

  // Ensure all threads hit completion
  cudaDeviceSynchronize();

  // Copy results back from GPU.
  int *h_dayHisto = (int *)malloc(sizeof(int) * timeHorizon);
  handleCudaErr(cudaMemcpy(h_dayHisto, d_dayHisto, sizeof(int) * timeHorizon, cudaMemcpyDeviceToHost));
  DEBUG(DB_CUDAMEM, "Moved kernel histogram back from device to host");

  // Manipulate results as necessary, stop timer, and output.
  duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  int daySum = 0;
  for (int i = 1; i < timeHorizon; ++i) {
    daySum += h_dayHisto[i] * i;
    //std::cout << "h_dayHisto[" << i << "]: " << h_dayHisto[i] << std::endl;
  }
  std::cout << "Mean day boundary hit: " << daySum / particles << std::endl;
  std::cout << "Duration: " << duration << std::endl;

  // Free device and host memory
  cudaFree(d_stdv);
  cudaFree(d_exreturn);
  cudaFree(d_shock);
  cudaFree(d_dayHisto);
  free(h_dayHisto);
  DEBUG(DB_CUDAMEM, "Successfully freed all device and host memory");
}
