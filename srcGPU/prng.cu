/*
 * CUDA for Prognostics - PRNG Implementation
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/

#include <iostream>
#include "curand.h"
#include "prng.h"
#include "debugCFP.h"

void prngUniformDouble (double *devStart, int count) {
  int status;
  curandGenerator_t gen;

  status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
  DEBUG(DB_CURAND, "Created a PRNG for uniformly distributed doubles.");
  status |= curandSetPseudoRandomGeneratorSeed(gen, 4294967296ULL^time(NULL));
  status |= curandGenerateUniformDouble(gen, devStart, count);
  status |= curandDestroyGenerator(gen);
  if (status != CURAND_STATUS_SUCCESS) {
    std::cout << "CuRand Failure!" << std::endl;
    exit(EXIT_FAILURE);
  }
  DEBUG(DB_CURAND, "Successfully generated doubles.");
}

void prngNormalDouble (double *devStart, int count, double mean, double stddev) {
  int status;
  curandGenerator_t gen;

  status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
  DEBUG(DB_CURAND, "Created a PRNG for normally distributed doubles.");
  status |= curandSetPseudoRandomGeneratorSeed(gen, 4294967296ULL^time(NULL));
  status |= curandGenerateNormalDouble(gen, devStart, count, mean, stddev);
  status |= curandDestroyGenerator(gen);
  if (status != CURAND_STATUS_SUCCESS) {
    std::cout << "CuRand Failure!" << std::endl;
    exit(EXIT_FAILURE);
  }
  DEBUG(DB_CURAND, "Successfully generated doubles.");
}
