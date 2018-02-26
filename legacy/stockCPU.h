/*
 * CUDA for Prognostics - Header
 *
 * Provides implementation of stockSim but using the CPU instead of GPU.
 * Used for time comparisons with the GPU as well as an example of
 * compiling the program for systems without a CUDA enabled device.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 31st, 2018
*/
#ifndef STOCKCPU_H_
#define STOCKCPU_H_

/* CPU version for running a monte carlo simulation of stock
 * prices over time. Prices all start at a specific fixed value and then
 * update on a daily basis until they hit a certain price (upper and lower
 * bounds) or the passed in time horizon.
*/
void runStockSim(int particles, int timeStep, int timeHorizon);

#endif
