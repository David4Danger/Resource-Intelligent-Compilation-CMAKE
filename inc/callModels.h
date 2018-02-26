/*
 * CUDA for Prognostics - Header
 *
 * Provides definitions of model wrapper functions. Launch dimensions are
 * currently determined at runtime via command line but this can be easily
 * modified here.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/
#ifndef CALLMODELS_H_
#define CALLMODELS_H_

/*********************************************************************/
/*           IMPORTANT NOTE ON CUDA MACROS AND THEIR USAGE           */
/*                                                                   */
/* In this module, two macros are used: __CUDA_ARCH__ and _CUDACC__  */
/* however they are used in two slightly different contexts. The     */
/* former of the two, __CUDA_ARCH__, should only be used to          */
/* differentiate between host and device compilation trajectories.   */
/* This means it should only be used in code quantified by           */
/* HOSTDEVICEQUALIFIER. The other macro, __CUDACC__, is used to      */
/* simply differentiate between CUDA and non-CUDA code. It should be */
/* used in situations in which is should be checked if nvcc is       */
/* steering compilation or not.                                      */
/*                                                                   */
/*********************************************************************/

/* This function makes a call to the stockModel function, which can either be
 * for CPU-only achitecture or alternatively the accelerated GPU version using
 * CUDA.
 *
 * Should be used as a sample/demo of how to modify existing models to support
 * dynamic compilation depending on the system used and it's available hardware.
*/
void callStockModel(int threadsPerBlock,
                    int blocksPerGrid,
                    int particles,
                    int timeStep,
                    int timeHorizon);
#endif
