/*
 * CUDA for Prognostics - Testing file
 *
 * Use this entry point to test out calls to the library, or copy it to
 * begin creation of another test file.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/
#include <iostream>
#include <cstdlib>
#include "callModels.h"


static void usage() {
  std::cout << "Didn't get enough parameters!" << std::endl <<
  "Usage: ./Progname threadsPerBlock blocksPerGrid particles timeStep timeHorizon ARGS"
  << std::endl << "Args:" << std::endl << "-dbg (CUDAMEM | CURAND | SYNC)"
  << std::endl;
  exit(EXIT_FAILURE);
}

/* Sample program handler. Invokes a model call with command line args.
 *
 * threadsPerBlock: Number of threads in a GPU block of execution
 * blocksPerGrid: Number of blocks in a GPU grid
 * particles: NUmber of particles used in the filter for simulation
 * timeStep: Interval of time between each step in the model
 * timeHorizon: Timeframe to test for an event in the model
 *
*/
int main (int argc, char** argv) {
  if (argc < 6) {
    usage();
  }

  int threadsPerBlock = atoi(argv[1]);
  int blocksPerGrid = atoi(argv[2]);
  int particles = atoi(argv[3]);
  int timeStep = atoi(argv[4]);
  int timeHorizon = atoi(argv[5]);

  //Invoke a model call function
  callStockModel(threadsPerBlock, blocksPerGrid, particles, timeStep, timeHorizon);

  return 0;
}
