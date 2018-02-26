/*
 * CUDA for Prognostics - Header
 *
 * Provides definitions of kernel wrapper functions. Launch dimensions are
 * currently determined at runtime via command line but this can be easily
 * modified here.
 *
 * Note that callkerns.cu, prng.cu and kernels.cu should be the only files
 * which require nvcc for compilation, all others can be done with standard.
 * g++. This header however must be included by any modules wishing to call
 * a CUDA code wrapper, and callskerns.o, kernels.o, prng.o must all be
 * included in the linking stage.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/
#ifndef CALLKERNS_H_
#define CALLKERNS_H_

/* Use this as a guide when writing new wrappers. Adjust as necessary.
 * Begin by removing the block comments for each section and removing the unecessary
 * ones. Ideally most of the template is used and only additions are made.
*/
void genericKernelWrapper(int threadsPerBlock,
                          int blocksPerGrid,
                          int particles,
                          int timeStep,
                          int timeHorizon);

/* This sample kernel call is for running a monte carlo simulation of stock
 * prices over time. Prices all start at a specific fixed value and then
 * update on a daily basis until they hit a certain price (upper and lower
 * bounds) or the passed in time horizon.
*/
void runStockSimKernel(int threadsPerBlock,
                       int blocksPerGrid,
                       int particles,
                       int timeStep,
                       int timeHorizon);

#endif
