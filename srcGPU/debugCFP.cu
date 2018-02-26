/*
 * CUDA for Prognostics - Implementation
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
 *
*/

#include "debugCFP.h"
#include <iostream>

int dbflags = 0;

void DEBUG(int flag, std::string msg) {
  if (dbflags & flag) {
    std::cout << msg << std::endl;
  }
}

void enableDbgFlag(std::string flag) {
  std::cout << "Enabling debug flag: " << flag << std::endl;
  if (flag == "CUDAMEM") {
    dbflags = dbflags | DB_CUDAMEM;
  } else if (flag == "CURAND") {
    dbflags = dbflags | DB_CURAND;
  } else if (flag == "SYNC") {
    dbflags = dbflags | DB_SYNC;
  } else {
    std::cout << "Recieved unknown flag: " << flag << std::endl;
    exit(EXIT_FAILURE);
  }
}

void handleCudaErr(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cout << "CUDA error occurred: " << cudaGetErrorString(err)
    << std::endl << "Exiting." << std::endl;
    exit(EXIT_FAILURE);
  }
}
