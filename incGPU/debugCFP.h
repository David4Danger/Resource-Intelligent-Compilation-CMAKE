/*
 * CUDA for Prognostics - Header
 * 
 * Defines debug functions used in the CFP framework.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/
#ifndef DEBUGCFP_H_
#define DEBUGCFP_H_

#include <string>

#define DB_CUDAMEM   0x0001
#define DB_CURAND    0x0002
#define DB_SYNC      0x0004

/* Used to enable a debug flag as indicated by command line.
Usage: enableDbgFlag(
*/
void enableDbgFlag(std::string flag);

/* Prints a debug message if the indicated flag is active.
Usage: DEBUG(DB_SOMEFLAG, "Performed an atomic add to variable foo");
*/
void DEBUG(int flag, std::string msg);

/* Used to check for errors when calling functions that involve device
memory, such as cudaMalloc, cudaMemcpy, etc.
Usage: handleDeviceMemErr(cudaMalloc(....))
*/
void handleCudaErr(cudaError_t err);

#endif
