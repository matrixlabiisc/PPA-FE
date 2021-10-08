

#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/laguerre.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <deal.II/grid/tria.h>

#include "mathUtils.h"
#include "matrixmatrixmul.h"
#include "atomicOrbitalBasisManager.h"
#include "bungeOrbitalInfo.h"

void AtomicOrbitalBasisManager::readBungeRadialBasisFunctions(){

	ROfBungeBasisFunctions = getRofBungeOrbitalBasisFuncs(atomType);
}


double AtomicOrbitalBasisManager::bungeOrbital
			(const OrbitalQuantumNumbers& orbital, 
			 const dealii::Point<3>& evalPoint, 
			 const std::vector<double>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);
		
	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return radialPartOfBungeOrbital(n, l, r)
			* realSphericalHarmonics(l, m, theta, phi);
}

double AtomicOrbitalBasisManager::bungeOrbital
			(const OrbitalQuantumNumbers& orbital, 
			 const dealii::Point<3>& evalPoint, 
			 const std::array<double, 3>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);

	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return radialPartOfBungeOrbital(n, l, r)
			* realSphericalHarmonics(l, m, theta, phi);
}

double AtomicOrbitalBasisManager::radialPartOfBungeOrbital
	(unsigned int n, unsigned int l, double r) {

	unsigned int azimHierarchy = n*(n-1)/2 + l;
	
	return ROfBungeBasisFunctions[azimHierarchy](r);
}



double AtomicOrbitalBasisManager::hydrogenicOrbital
			(const OrbitalQuantumNumbers& orbital, 
			 const dealii::Point<3>& evalPoint, 
			 const std::vector<double>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);
		
	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return radialPartOfHydrogenicOrbital(n, l, r)
			* realSphericalHarmonics(l, m, theta, phi);
}

double AtomicOrbitalBasisManager::hydrogenicOrbital
			(const OrbitalQuantumNumbers& orbital, 
			 const dealii::Point<3>& evalPoint, 
			 const std::array<double, 3>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);

	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return radialPartOfHydrogenicOrbital(n, l, r)
			* realSphericalHarmonics(l, m, theta, phi);
}

double AtomicOrbitalBasisManager::radialPartOfHydrogenicOrbital
	(unsigned int n, unsigned int l, double r) {

	double tmp1 = 2*zeta/n;
	double tmp2 = tmp1 * r; 

	// double r = distance3d(evalPoint, atomPos); 

	return tmp1 * sqrt(tmp1 * factorial(n-l-1)/(2.0*n*factorial(n+l))) *
		   boost::math::laguerre(n-l-1, 2*l+1, tmp2) *
		   pow(tmp2, l) *
		   exp(-tmp2/2);
}

double AtomicOrbitalBasisManager::slaterTypeOrbital
		(const OrbitalQuantumNumbers& orbital, 
		 const dealii::Point<3>& evalPoint, 
		 const std::vector<double>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);

	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return radialPartofSlaterTypeOrbital(n, r)
			* realSphericalHarmonics(l, m, theta, phi);
}

double AtomicOrbitalBasisManager::slaterTypeOrbital
		(const OrbitalQuantumNumbers& orbital, 
		 const dealii::Point<3>& evalPoint, 
		 const std::array<double, 3>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);

	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return radialPartofSlaterTypeOrbital(n, r)
			* realSphericalHarmonics(l, m, theta, phi);
}

double AtomicOrbitalBasisManager::radialPartofSlaterTypeOrbital
	(unsigned int n, double r){

	double tmp1 = 2*zeta/n;
	double tmp2 = tmp1 * r/2;

	// equivalent to: (just for readability)
	// double normalizationConst = pow(2*zeta/n, n) * sqrt(2*zeta/(n*factorial(2*n))); 
	// return normalizationConst * pow(r, n-1) * exp(-zeta*r/n); 

	double normalizationConst = pow(tmp1, n) * sqrt(tmp1/factorial(2*n));

	return normalizationConst * pow(r, n-1) * exp(-tmp2); 
}


// https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Spherical_harmonics
// https://en.wikipedia.org/wiki/Spherical_harmonics

double AtomicOrbitalBasisManager::realSphericalHarmonics
	(unsigned int l, short int m, double theta, double phi){

	double sphericalHarmonicVal{}; 

	// sphericalHarmonicVal = boost::math::spherical_harmonic_i(0, 0, 0.0, 0.0);
	// even this gives the same error 
	// can replace std::sqrt(2.0) with M_SQRT2

	// https://en.wikipedia.org/wiki/Spherical_harmonics
	// in quantum mechanics there is an additional (-1)^m
	// which is not present in the below boost function which we are using
	// https://www.boost.org/doc/libs/1_76_0/libs/math/doc/html/math_toolkit/sf_poly/sph_harm.html

	// may be try: https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c
	// https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
	// may be some specific solution like m % 2 == 0 then 1 or -1 
	// -2 * (m % 2) + 1 seems good :) 

	if (m < 0) {
      	sphericalHarmonicVal = pow(-1, m) *
      		M_SQRT2 * boost::math::spherical_harmonic_i(l, -m, theta, phi);
	}

	else if (m == 0) {
	    sphericalHarmonicVal =
	        boost::math::spherical_harmonic_r(l, m, theta, phi);
	}

	else if (m > 0) {
	    sphericalHarmonicVal = pow(-1, m) * 
	        M_SQRT2 * boost::math::spherical_harmonic_r(l, m, theta, phi);
	}

  	return sphericalHarmonicVal;
}

// for the 1S orbital of each hydrogen atom  
// could also have made the radial part static member function 

double radialPartofSlaterTypeOrbitalTest
	(unsigned int n, const dealii::Point<3>& evalPoint, const std::vector<double>& atomPos){

	// double normalizationConst = pow(2, n) * pow(zeta, n) * sqrt(2*zeta/factorial(2*n)); 
	// pow(2, n) might be expensive, we have have a memory or use 1 << n, check which is better
	// and for factorial we can also use memory: factorial memory function - lookup table!  
	// usually we would expect upto only 10 basis functions 
	// above that memorization (brute force form of memoization) would be helpful

	double zeta = 1.3;

	double r = distance3d(evalPoint, atomPos); 

	double normalizationConst = (1 << n) * pow(zeta/n, n) * sqrt(2*zeta/(n*factorial(2*n))); 

	return normalizationConst * pow(r, n-1) * exp(-zeta*r/n); 
}

double radialPartofSlaterTypeOrbitalTest
	(unsigned int n, const dealii::Point<3>& evalPoint, const std::array<double, 3>& atomPos){

	// double normalizationConst = pow(2, n) * pow(zeta, n) * sqrt(2*zeta/factorial(2*n)); 
	// pow(2, n) might be expensive, we have have a memory or use 1 << n, check which is better
	// and for factorial we can also use memory: factorial memory function - lookup table!  
	// usually we would expect upto only 10 basis functions 
	// above that memorization (brute force form of memoization) would be helpful

	double zeta = 1.3;

	double r = distance3d(evalPoint, atomPos); 

	double normalizationConst = (1 << n) * pow(zeta/n, n) * sqrt(2*zeta/(n*factorial(2*n))); 

	return normalizationConst * pow(r, n-1) * exp(-zeta*r/n); 
}


double hydrogenMoleculeBondingOrbital(const dealii::Point<3>& evalPoint)
{
	const std::array<double, 3> atomPos1{-0.69919867, 0.0, 0.0};
	const std::array<double, 3> atomPos2{0.69919867, 0.0, 0.0};

	double phi1 = radialPartofSlaterTypeOrbitalTest
					(1, evalPoint, atomPos1) * sqrt(1/(4*M_PI));
	double phi2 = radialPartofSlaterTypeOrbitalTest
					(1, evalPoint, atomPos2) * sqrt(1/(4*M_PI));

	const double s = 0.63634108;

	return (phi1 + phi2)/sqrt(2*(1 + s)); // forgot the 1+s part 
}
