/*
 * CUDA for Prognostics - stockSim Module
 *
 * Implements stockSim using stictly the CPU
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 31st, 2018
*/

#include "stockCPU.h"
#include <random>
#include <iostream>
#include <ctime>

/* Basic monte carlo simulation for calculating stock price. See header file
 * for additional details.
*/
void runStockSim(int particles, int timeStep, int timeHorizon) {
  // Declare variables. Note that standard deviation and expected return are
  // randomly generated below.
  double stockPrice = 20.00;
  double stockMin = 3.75;
  double stockMax = 324.25;
  double *stdv;
  double *exreturn;
  // Next variable used to determine if shock is + or -; need particles * timeH
  double *shock;
  int *dayHisto;

  // Allocate and populate input arrays for device and output histogram.
  stdv = (double *)malloc(sizeof(double) * particles);
  exreturn = (double *)malloc(sizeof(double) * particles);
  shock = (double *)malloc(sizeof(double) * particles * timeHorizon);
  dayHisto = (int *)calloc(timeHorizon, sizeof(double));

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

  // Begin timer
  std::clock_t start;
  double duration;
  start = std::clock();

  // Simulate
  double drift, shockVal, curPrice, deltaPrice;
  int t;
  for (int i = 0; i < particles; ++i) {
    curPrice = stockPrice;
    drift = exreturn[i] * 0.1 * (double)timeStep;
    shockVal = stdv[i] * 0.1 * sqrt((double)timeStep);

    for (t = 0; t < timeHorizon; t+= timeStep) {
      // simulate the model forward one step
      if (shock[t * i] < 0.50) {
        deltaPrice = curPrice * (drift - shockVal);
      } else {
        deltaPrice = curPrice * (drift + shockVal);
      }

      curPrice += deltaPrice;//update the price

      // Check if stock price exceeds a boundary
      bool boundaryPassed = (curPrice > stockMax) || (curPrice < stockMin);
      if (boundaryPassed) {
        dayHisto[t] += 1;
        break;
      }
    }

    if (t >= timeHorizon) {
      // Never broke boundary, just add to final day
      dayHisto[timeHorizon - timeStep] += 1;
    }
  }

  // Manipulate results as necessary, stop timer, and output.
  duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  int daySum = 0;
  for (int i = 0; i < timeHorizon; ++i) {
    daySum += dayHisto[i] * i;
    //std::cout << "dayHisto[" << i << "]: " << dayHisto[i] << std::endl;
  }
  std::cout << "Mean day boundary hit: " << daySum / particles << std::endl;
  std::cout << "Duration: " << duration << std::endl;

  // Free device and host memory
  free(stdv);
  free(exreturn);
  free(shock);
  free(dayHisto);
}
