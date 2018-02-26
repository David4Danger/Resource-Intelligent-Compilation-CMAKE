/*
 * CUDA for Prognostics - Header
 *
 * Provides definitions of kernel calls.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/
#ifndef KERNELS_H_
#define KERNELS_H_

/* Use this as a template for kernel calls. Many possible optimizations exist
 * on a per-kernel basis, based on the nature of the program.
*/
__global__ void genericKernelCall(int particles, int timeStep, int timeHorizon);

/* This kernel simulates a single particle for stock price over time. It
 * uses a model that assumes that for each timeStep, the stock price will
 * "drift up" by the expected return rate. However the drift will be 'shocked'
 * (added or subtracted by a random amount) for each timeStep.
 * The simulation for each particle ends when it either reaches the timeHorizon
 * or hits the upper or lower stock price limit.
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
                               int *dayHisto);

#endif
