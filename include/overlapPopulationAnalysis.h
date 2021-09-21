#pragma once
/*
*	
*	The following functions are used for post-processing the
*	projections of Kohn-Sham orbitals onto chosen atomic orbitals
*	 
*	We construct pCOOP, pCOHP, spillFactors, projectabilities, pDOS's
*
*
*/

#ifndef OVERLAP_POPULATIONS_H_
#define OVERLAP_POPULATIONS_H_

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <valarray>
#include "distributions.h"


template <typename T>
void printVector(const std::vector<T>& vec)
{
	for (const auto& elem : vec)
	{
		std::cout << elem << ' ';
	}
	std::cout << '\n';
}

template <typename T, std::size_t N>
void printArray(const std::array<T, N>& array)
{
	for (const auto& elem : array)
	{
		std::cout << elem << ' ';
	}
	std::cout << '\n';
}

struct spillFactors {

	double absChargeSpilling;
	double absTotalSpilling;
	double absOccupiedBandsSpilling;

	double chargeSpilling;
	double totalSpilling;
	double occupiedBandsSpilling;

	std::vector<double> projectabilities;

	// note that total spilling in both variants is the spill factor for all bands 
	// and charge spilling in both variants is the spill factor for occupied bands 
	// later we can also check what projectability is and its relation to spill factor 
	// occupiedBandsSpills average over only the occupied orbitals 
};



// this function assumes all filled Kohn-Sham orbitals/bands come first
// to find the first zero in the occupation number vector  
unsigned int numberOfFilledBands(const std::vector<double>& occupationNum);


spillFactors spillFactorsOfProjection(const std::vector<double>& coeffMatrixVecOfProj,
									  const std::vector<double>& arrayVecOfProj,
									  const std::vector<double>& occupationNum);


std::vector<double> 
pCOOPvsEnergy(std::vector<double> epsvalues,
			  int globalBasisNum1, int globalBasisNum2, 
			  const std::vector<double>&  SmatrixVec,
			  const std::vector<double>& coeffArrayOfProj,
			  const std::vector<double>& energyLevelsKS,
			  const std::vector<int>& occupationNum,
			  std::vector<double>& pCOOPcoeffs );


// returns pCOHPvalues correspoding energy values vector 
std::vector<double> 
pCOHPvsEnergyTest(std::vector<double> epsvalues,
			  int globalBasisNum1, int globalBasisNum2, // a, b
			  const std::vector<double>&  HmatrixVec,
			  const std::vector<double>& coeffArrayOfProj,
			  const std::vector<double>& energyLevelsKS,
			  const std::vector<int>& occupationNum,
			  std::vector<double>& pCOHPcoeffs );



#endif

