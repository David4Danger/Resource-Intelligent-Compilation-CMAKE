/*
 * CUDA for Prognostics - Header
 *
 * Takes care of random number generation on CUDA devices.
 * Much more efficient than generating numbers on the host and copying them over
 * because a) no memory transfer is required and b) the random number generation
 * can be done in parallel.
 *
 * Created by: David Skudra (david.skudra@nasa.gov)
 * Created on: Jan 23rd, 2018
*/
#ifndef PRNG_H_
#define PRNG_H_

// Allocates count uniform doubles using the mersenne random number generator
// starting at address devStart (a device memory address)
void prngUniformDouble (double *devStart, int count);

// Allocates count normal doubles using the mersenne random number generator
// starting at address devStart (a device memory address). Most simulations
// are better off using a uniform distribution.
void prngNormalDouble (double *devStart, int count);

#endif
