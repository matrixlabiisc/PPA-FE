

#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/laguerre.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <deal.II/grid/tria.h>

#include "mathUtils.h"
#include "matrixmatrixmul.h"
#include "atomicOrbitalBasisManager.h"

/** @brief Normalized Radial part of Slater Type Orbital 
 * 
 * 
 * 
 */
double AtomicOrbitalBasisManager::ROfSTO(unsigned int n, double zetaEff, double r){

    double tmp = 2*zetaEff;

    double normalizationConst = pow(tmp, n) * sqrt(tmp/factorial(2*n));

    return normalizationConst * pow(r, n-1) * exp(-zetaEff*r); 
}

/** @brief Returns the Bunge orbital basis functions for a given Atomic number
 * 
 *  
 * 
 * 
 * 
 * 
 * 
 */
std::vector< std::function<double(double)> >
AtomicOrbitalBasisManager::getRofBungeOrbitalBasisFuncs(unsigned int atomicNum){

    std::vector< std::function<double(double)> > bungeFunctions;

    // properties of the given data 
    unsigned int nMax; 
    unsigned int nMin;
    unsigned int lMax;
    unsigned int numOfOrbitals;

	switch(atomicNum) {

        case 6: // Carbon 

            nMin = 1;
            nMax = 2;
            lMax = 1;
            numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([](double r){return 
                
                + 0.352872 * ROfSTO(1, 8.4936, r)
                + 0.473621 * ROfSTO(1, 4.8788, r)
                - 0.001199 * ROfSTO(3, 15.466, r)
                + 0.210887 * ROfSTO(2, 7.0500, r)
                + 0.000886 * ROfSTO(2, 2.2640, r)
                + 0.000465 * ROfSTO(2, 1.4747, r)
                - 0.000119 * ROfSTO(2, 1.1639, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([](double r){return 
                
                - 0.071727 * ROfSTO(1, 8.4936, r)
                + 0.438307 * ROfSTO(1, 4.8788, r)
                - 0.000383 * ROfSTO(3, 15.466, r)
                - 0.091194 * ROfSTO(2, 7.0500, r)
                - 0.393105 * ROfSTO(2, 2.2640, r)
                - 0.579121 * ROfSTO(2, 1.4747, r)
                - 0.126067 * ROfSTO(2, 1.1639, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([](double r){return 
                
                + 0.006977 * ROfSTO(2, 7.0500, r)
                + 0.070877 * ROfSTO(2, 3.2275, r)
                + 0.230802 * ROfSTO(2, 2.1908, r)
                + 0.411931 * ROfSTO(2, 1.4413, r)
                + 0.350701 * ROfSTO(2, 1.0242, r);
            });

            break;

        case 8: // Oxygen

            nMin = 1;
            nMax = 2;
            lMax = 1;
            numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([](double r){return 
                
                + 0.360063 * ROfSTO(1, 11.2970, r)
                + 0.466625 * ROfSTO(1, 6.5966, r)
                - 0.000918 * ROfSTO(3, 20.5019, r)
                + 0.208441 * ROfSTO(2, 9.5546, r)
                + 0.002018 * ROfSTO(2, 3.2482, r)
                + 0.000216 * ROfSTO(2, 2.1608, r)
                + 0.000133 * ROfSTO(2, 1.6411, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([](double r){return 
                
                - 0.064363 * ROfSTO(1, 11.2970, r)
                + 0.433186 * ROfSTO(1, 6.5966, r)
                - 0.000275 * ROfSTO(3, 20.5019, r)
                - 0.072497 * ROfSTO(2, 9.5546, r)
                - 0.369900 * ROfSTO(2, 3.2482, r)
                - 0.512627 * ROfSTO(2, 2.1608, r)
                - 0.227421 * ROfSTO(2, 1.6411, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([](double r){return 
                
                + 0.005626 * ROfSTO(2, 9.6471, r)
                + 0.126618 * ROfSTO(2, 4.3323, r)
                + 0.328966 * ROfSTO(2, 2.7502, r)
                + 0.395422 * ROfSTO(2, 1.7525, r)
                + 0.231788 * ROfSTO(2, 1.2473, r);
            });

            break;

        default:

            std::cout << "Bunge orbital data not filled for "
                      << "atomic number: " << atomicNum << '\n';

            std::exit(-1);
    }

    return bungeFunctions;
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
