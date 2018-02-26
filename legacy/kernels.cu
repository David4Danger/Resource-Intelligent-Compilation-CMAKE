/*
 * CUDA for Prognostics - Kernel body file
 *
 * All kernel code should be exclusively implemented here.
 * Ideally the state and threshold equations exist in their own module.
 *
 * Also note that constant memory won't really be useful here, as each
 * particle is independent. As such, the broadcast + caching features of
 * constant memory won't be useful.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/

#include <cmath>
#include "kernels.h"
#include "callkerns.h"
//Unfortunately the DEBUG functionality is CPU only, can't be used here.

/* Use this as a template for kernel calls. Many possible optimizations exist
 * on a per-kernel basis, based on the nature of the program.
*/
__global__ void genericKernelCall(int particles,
                                  int timeStep,
                                  int timeHorizon) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize all variables needed for model

  // Cover all particles mapped to this id and strides
  while (id < particles) {
    for (int t = 0; t < timeHorizon; ++timeStep) {
      // simulate the model forward one step
      /*SOME STATE EQN HERE*/

      // Check for some event
      /*bool eventOccured = THRESHOLD EQN HERE
      if (eventOccured) {
        UPDATE RESULT ARRAY i.e arr[id] = t;
        break;
      }*/
    }

    id += stride;
  }

  //OPTIONAL: Reduction performed in parallel here, in place of on CPU
}

////////////////////////////////////////////////////////////////////////////////

/* Kernel call of a basic monte carlo stock price simulation. Ideally launched with
 * warp threadsPerBlock, and the maximum number of CUDA cores on the system. See
 * more details in header.
 */
__global__ void stockSimKernel(int particles,
                               int timeStep,
                               int timeHorizon,
                               double stockPrice,
                               double stockMin,
                               double stockMax,
                               double *stdv,
                               double *exreturn,
                               double *shockBools,
                               int *dayHisto) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize all variables needed for model
  double drift, shockVal, curPrice, deltaPrice;

  // Cover all particles mapped to this id and strides
  while (id < particles) {
    int t;
    curPrice = stockPrice;//this thread might need stockPrice again
    drift = exreturn[id] * 0.1 * (double)timeStep;
    shockVal = stdv[id] * 0.1 * sqrt((double)timeStep);

    for (t = 0; t < timeHorizon; t += timeStep) {
      // simulate the model forward one step
      if (shockBools[t * id] < 0.50) {
        deltaPrice = curPrice * (drift - shockVal);
      } else {
        deltaPrice = curPrice * (drift + shockVal);
      }

      curPrice += deltaPrice;//update the price

      // Check if stock price exceeds a boundary
      bool boundaryPassed = (curPrice > stockMax) || (curPrice < stockMin);
      if (boundaryPassed) {
        atomicAdd(&dayHisto[t], 1);//no race conditions here folks
        break;
      }
    }

    if (t >= timeHorizon) {
      // Never broke boundary, just add to final day
      atomicAdd(&dayHisto[timeHorizon - timeStep], 1);
    }
    id += stride;
  }
}
