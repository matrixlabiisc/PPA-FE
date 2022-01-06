

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <valarray>
#include "distributions.h"
#include "overlapPopulationAnalysis.h"
#include "matrixmatrixmul.h"


// this function assumes all filled Kohn-Sham orbitals/bands come first
// to find the first zero in the occupation number vector  
unsigned int numberOfFilledBands(const std::vector<double>& occupationNum){

	unsigned int numOfFilledKSorbitals
		= std::distance(std::begin(occupationNum), 
						std::find_if(std::begin(occupationNum), std::end(occupationNum), 
						[](double x) { return (std::abs(x) < 1e-05); }));

	return numOfFilledKSorbitals; 
}


spillFactors spillFactorsOfProjection(const std::vector<double>& coeffMatrixVecOfProj,
									  const std::vector<double>& arrayVecOfProj,
									  const std::vector<double>& occupationNum){

	spillFactors spillvalues = {};

	unsigned int numOfFilledKSorbitals = numberOfFilledBands(occupationNum);
	unsigned int numOfKSOrbitals = occupationNum.size();
	unsigned int totalDimOfBasis = arrayVecOfProj.size()/numOfKSOrbitals;
	double totalNumOfElectrons = 0.0; 

	spillvalues.projectabilities.resize(numOfKSOrbitals, 0.0);

	for(auto &n : occupationNum){ totalNumOfElectrons += n; }

        totalNumOfElectrons = 2*totalNumOfElectrons;

	unsigned int index; 

	std::valarray<double> spillForEachBand(1.0, numOfKSOrbitals); 
	std::valarray<double> chargeSpillForEachBand(0.0, numOfFilledKSorbitals);

	for (size_t i = 0; i < numOfKSOrbitals; ++i) 
	{
		for (size_t k = 0; k < totalDimOfBasis; ++k)
		{
			index = i + k*numOfKSOrbitals;

			spillvalues.projectabilities[i] += coeffMatrixVecOfProj[index] * 
										 	   arrayVecOfProj[index];
		}

		spillForEachBand[i] -= spillvalues.projectabilities[i];

		if(i < numOfFilledKSorbitals){

			chargeSpillForEachBand[i] = 2 * spillForEachBand[i] * occupationNum[i];

			spillvalues.occupiedBandsSpilling += spillForEachBand[i];
			spillvalues.absOccupiedBandsSpilling += std::abs(spillForEachBand[i]);
		}
	}

	spillvalues.occupiedBandsSpilling /= numOfFilledKSorbitals;
	spillvalues.absOccupiedBandsSpilling /= numOfFilledKSorbitals;

	// actually we could have summed up everything in the above loop itself to save space

	spillvalues.totalSpilling = spillForEachBand.sum()/numOfKSOrbitals;

	spillvalues.chargeSpilling = chargeSpillForEachBand.sum()/
										(numOfFilledKSorbitals * totalNumOfElectrons);

	spillvalues.absTotalSpilling = std::abs(spillForEachBand).sum()/numOfKSOrbitals;

	spillvalues.absChargeSpilling = std::abs(chargeSpillForEachBand).sum()/
										(numOfFilledKSorbitals * totalNumOfElectrons);
	
	return spillvalues;
}


void spillFactorsofProjectionwithCS(const std::vector<double> & C,
									const std::vector<double> & Sold,
									const std::vector<double> & occupationNum,
									int m1, int n1, int m2, int n2)
{
	int N = n1;
	std::vector<double> S(m2*n2,0.0);
	int count = 0;
	for(int i = 0; i < m2; i++)
	{
		for(int j = i; j < n2; j++)
		{
			S[i*n1+j] = Sold[count];
			S[j*m1+i] = Sold[count];
			count++;
		}
	}
	auto temp = matrixTmatrixmul(C,n1,m1,S,m2,n2);
	auto O	  = matrixmatrixmul(temp,n1,n2,C,n1,m1);
	double TSF = 0.0;
	double CSF = 0.0;
	double TSFabs = 0.0;
	double CSFabs = 0.0;
	double totalNumOfElectrons = 0.0;
		unsigned int numOfFilledKSorbitals = std::distance(std::begin(occupationNum), 
											 std::find_if(std::begin(occupationNum), std::end(occupationNum), 
						                     [](double x) { return (std::abs(x) < 1e-05); }));
		unsigned int numOfKSOrbitals = occupationNum.size();
		for(auto &n : occupationNum){ totalNumOfElectrons += n; }

        totalNumOfElectrons = 2*totalNumOfElectrons;	
	for(int i = 0; i < N; i++)
	{
		TSF += 1- O[i*N+i];
		TSFabs += std::fabs(1- O[i*N+i]);	
		if(i < numOfFilledKSorbitals)
		{
			CSF += 2*(1- O[i*N+i]);
			CSFabs += 2*(std::fabs(1- O[i*N+i]));
		}
	}
	TSF /= N;
	TSFabs /=N;
	CSF /=(numOfFilledKSorbitals * totalNumOfElectrons);
	CSFabs /=(numOfFilledKSorbitals * totalNumOfElectrons);
	std::cout<<"TSF: "<<TSF<<std::endl;
	std::cout<<"TSFabs: "<<TSF<<std::endl;
	std::cout<<"CSF: "<<CSF<<std::endl;
	std::cout<<"CSFabs: "<<CSFabs<<std::endl;



}

std::vector<double> 
pCOOPvsEnergy(std::vector<double> epsvalues,
			  int globalBasisNum1, int globalBasisNum2, 
			  const std::vector<double>&  SmatrixVec,
			  const std::vector<double>& coeffArrayOfProj,
			  const std::vector<double>& energyLevelsKS,
			  const std::vector<int>& occupationNum,
			  std::vector<double>& pCOOPcoeffs ) {

	int numOfKSOrbitals = energyLevelsKS.size();
	int numOfdatapoints = epsvalues.size();
		
	int totalDimOfBasis = coeffArrayOfProj.size()/numOfKSOrbitals;

	int minindex = std::min(globalBasisNum1, globalBasisNum2);
	int maxindex = std::max(globalBasisNum1, globalBasisNum2);

	int index = totalDimOfBasis*minindex - minindex*(minindex + 1)/2 + maxindex;
	double S_ab = SmatrixVec[index];

	// std::cout << "Starting pCOOP vs energy calculation\n";

	std::vector<double> pCOOPvalues(numOfdatapoints, 0.0);

	// std::cout << "pCOOPvalues vector constructed\n";

	for(int j = 0; j < numOfKSOrbitals; ++j){

		// std::cout << "entered the loop\n";
		// std::cout << "occupationNum = " << occupationNum[j] << '\n';

		if(occupationNum[j] == 0) { continue; } // { break; }

		// may be use continue instead of break 
		// to see if there any further filled orbitals  

		// std::cout << "Starting pCOOP vs energy calculation\n";

		pCOOPcoeffs[j] = coeffArrayOfProj[j + globalBasisNum1 * totalDimOfBasis]*
			       		 coeffArrayOfProj[j + globalBasisNum2 * totalDimOfBasis]* 
			             S_ab * occupationNum[j];

		/* For testing H2 molecule case if everythng works fine 

		std::cout << S_ab << '\n';
		std::cout << coeffArrayOfProj[j][globalBasisNum1] << '\n';
		std::cout << coeffArrayOfProj[j][globalBasisNum2] << '\n';
		std::cout << occupationNum[j] << '\n';

		std::cout << "pCOOPcoeff: " << tmpcoeff << '\n';

		*/

		// std::cout << "Starting pCOOP vs energy calculation continues \n";

		for(int i = 0; i < numOfdatapoints; ++i){

			pCOOPvalues[i] += pCOOPcoeffs[j] * 
							  lorentzian(epsvalues[i], energyLevelsKS[j], 0.1);
		}

		std::cout << "pCOOP contribution of " << j << "th KSOrbital calculated!\n";

	}

	return pCOOPvalues;
}
// can use auto pCOOPvector to collect the return vector 


// returns pCOHPvalues correspoding energy values vector 
std::vector<double> 
pCOHPvsEnergyTest(std::vector<double> epsvalues,
			  int globalBasisNum1, int globalBasisNum2, // a, b
			  const std::vector<double>&  HmatrixVec,
			  const std::vector<double>& coeffArrayOfProj,
			  const std::vector<double>& energyLevelsKS,
			  const std::vector<int>& occupationNum,
			  std::vector<double>& pCOHPcoeffs ) {

	int numOfKSOrbitals = energyLevelsKS.size();
	int numOfdatapoints = epsvalues.size();
		
	int totalDimOfBasis = coeffArrayOfProj.size()/numOfKSOrbitals;

	int minindex = std::min(globalBasisNum1, globalBasisNum2);
	int maxindex = std::max(globalBasisNum1, globalBasisNum2);

	int index = totalDimOfBasis*minindex - minindex*(minindex + 1)/2 + maxindex;
	double H_ab = HmatrixVec[index];

	// double tmpcoeff;

	// std::cout << "Starting pCOHP vs energy calculation\n";

	std::vector<double> pCOHPvalues(numOfdatapoints, 0.0);

	// std::cout << "pCOHPvalues vector constructed\n";

	for(int j = 0; j < numOfKSOrbitals; ++j){

		// std::cout << "entered the loop\n";
		// std::cout << "occupationNum = " << occupationNum[j] << '\n';

		if(occupationNum[j] == 0) { continue; } // { break; }

		// may be use continue instead of break 
		// to see if there any further filled orbitals  

		// std::cout << "Starting pCOHP vs energy calculation\n";

		pCOHPcoeffs[j] = coeffArrayOfProj[j + globalBasisNum1 * totalDimOfBasis]*
			       		 coeffArrayOfProj[j + globalBasisNum2 * totalDimOfBasis]* 
			             H_ab * occupationNum[j];

		/* For testing H2 molecule case if everythng works fine 

		std::cout << H_ab << '\n';
		std::cout << coeffArrayOfProj[j][globalBasisNum1] << '\n';
		std::cout << coeffArrayOfProj[j][globalBasisNum2] << '\n';
		std::cout << occupationNum[j] << '\n';

		std::cout << "pCOHPcoeff: " << tmpcoeff << '\n';

		*/

		// std::cout << "Starting pCOHP vs energy calculation continues \n";

		for(int i = 0; i < numOfdatapoints; ++i){

			pCOHPvalues[i] += pCOHPcoeffs[j] * 
							  lorentzian(epsvalues[i], energyLevelsKS[j], 0.1);
		}

		std::cout << "pCOHP contribution of " << j << "th KSOrbital calculated!\n";

	}

	return pCOHPvalues;
}
// can use auto pCOHPvector to collect the return vector 





