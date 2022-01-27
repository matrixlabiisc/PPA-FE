

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
double AtomicOrbitalBasisManager::RofSTO(unsigned int n, double zetaEff, double r){

    double tmp = 2*zetaEff;

    double normalizationConst = pow(tmp, n) * sqrt(tmp/factorial(2*n));

    return normalizationConst * pow(r, n-1) * exp(-zetaEff*r); 
}

/** @brief Normalized Radial part of Hydrogenic Orbitals 
 * 
 * 
 * 
 */
double AtomicOrbitalBasisManager::RofHydrogenicOrbital
	(unsigned int n, unsigned int l, double zetaEff, double r) {

	double tmp1 = 2*zetaEff/n;
	double tmp2 = tmp1 * r; 

	return tmp1 * sqrt(tmp1 * factorial(n-l-1)/(2.0*n*factorial(n+l))) *
		   boost::math::laguerre(n-l-1, 2*l+1, tmp2) *
		   pow(tmp2, l) *
		   exp(-tmp2/2);
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

		case 1: // Hydrogen

			nMin = 1;
			nMax = 2;
			lMax = 1;
			numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // zeta value is NOT taken from the STOBasisInfo.inp file input 

            // 1s radial part of Hydrogen
            bungeFunctions.push_back([&](double r){return 
            	RofHydrogenicOrbital(1, 0, 1.0, r);});

            // 2s radial part of Hydrogen
            bungeFunctions.push_back([&](double r){return 
            	RofHydrogenicOrbital(2, 0, 1.0, r);});
            
            // 2p radial part of Hydrogen 
            bungeFunctions.push_back([&](double r){return 
            	RofHydrogenicOrbital(2, 1, 1.0, r);});

            break; 

        case 3: // Lithium 

            nMin = 1;
            nMax = 2;
            lMax = 0;
            numOfOrbitals = 2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.141279 * RofSTO(1, 4.3069, r)
                + 0.874231 * RofSTO(1, 2.4573, r)
                - 0.005201 * RofSTO(3, 6.7850, r)
                - 0.002307 * RofSTO(2, 7.4527, r)
                + 0.006985 * RofSTO(2, 1.8504, r)
                - 0.000305 * RofSTO(2, 0.7667, r)
                + 0.000760 * RofSTO(2, 0.6364, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                - 0.022416 * RofSTO(1, 4.3069, r)
                - 0.135791 * RofSTO(1, 2.4573, r)
                + 0.000389 * RofSTO(3, 6.7850, r)
                - 0.000068 * RofSTO(2, 7.4527, r)
                - 0.076544 * RofSTO(2, 1.8504, r)
                + 0.340542 * RofSTO(2, 0.7667, r)
                + 0.715708 * RofSTO(2, 0.6364, r);
            });

            break;

        case 4: // Beryllium 

            nMin = 1;
            nMax = 2;
            lMax = 0;
            numOfOrbitals = 2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.285107 * RofSTO(1, 5.7531, r)
                + 0.474813 * RofSTO(1, 3.7156, r)
                - 0.001620 * RofSTO(3, 9.9670, r)
                + 0.052852 * RofSTO(3, 3.7128, r)
                + 0.243499 * RofSTO(2, 4.4661, r)
                + 0.000106 * RofSTO(2, 1.2919, r)
                - 0.000032 * RofSTO(2, 0.8555, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                - 0.016378 * RofSTO(1, 5.7531, r)
                - 0.155066 * RofSTO(1, 3.7156, r)
                + 0.000426 * RofSTO(3, 9.9670, r)
                - 0.059234 * RofSTO(3, 3.7128, r)
                - 0.031925 * RofSTO(2, 4.4661, r)
                + 0.387968 * RofSTO(2, 1.2919, r)
                + 0.685674 * RofSTO(2, 0.8555, r);
            });

            break;

        case 5: // Boron 

            nMin = 1;
            nMax = 2;
            lMax = 1;
            numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.381607 * RofSTO(1, 7.0178, r)
                + 0.423958 * RofSTO(1, 3.9468, r)
                - 0.001316 * RofSTO(3, 12.7297, r)
                - 0.000822 * RofSTO(3, 2.7646, r)
                + 0.237016 * RofSTO(2, 5.7420, r)
                + 0.001062 * RofSTO(2, 1.5436, r)
                - 0.000137 * RofSTO(2, 1.0802, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                - 0.022549 * RofSTO(1, 7.0178, r)
                + 0.321716 * RofSTO(1, 3.9468, r)
                - 0.000452 * RofSTO(3, 12.7297, r)
                - 0.072032 * RofSTO(3, 2.7646, r)
                - 0.050313 * RofSTO(2, 5.7420, r)
                - 0.484281 * RofSTO(2, 1.5436, r)
                - 0.518986 * RofSTO(2, 1.0802, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                + 0.007600 * RofSTO(2, 5.7416, r)
                + 0.045137 * RofSTO(2, 2.6341, r)
                + 0.184206 * RofSTO(2, 1.8340, r)
                + 0.394754 * RofSTO(2, 1.1919, r)
                + 0.432795 * RofSTO(2, 0.8494, r);
            });

            break;

        case 6: // Carbon 

            nMin = 1;
            nMax = 2;
            lMax = 1;
            numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.352872 * RofSTO(1, 8.4936, r)
                + 0.473621 * RofSTO(1, 4.8788, r)
                - 0.001199 * RofSTO(3, 15.466, r)
                + 0.210887 * RofSTO(2, 7.0500, r)
                + 0.000886 * RofSTO(2, 2.2640, r)
                + 0.000465 * RofSTO(2, 1.4747, r)
                - 0.000119 * RofSTO(2, 1.1639, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                - 0.071727 * RofSTO(1, 8.4936, r)
                + 0.438307 * RofSTO(1, 4.8788, r)
                - 0.000383 * RofSTO(3, 15.466, r)
                - 0.091194 * RofSTO(2, 7.0500, r)
                - 0.393105 * RofSTO(2, 2.2640, r)
                - 0.579121 * RofSTO(2, 1.4747, r)
                - 0.126067 * RofSTO(2, 1.1639, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                + 0.006977 * RofSTO(2, 7.0500, r)
                + 0.070877 * RofSTO(2, 3.2275, r)
                + 0.230802 * RofSTO(2, 2.1908, r)
                + 0.411931 * RofSTO(2, 1.4413, r)
                + 0.350701 * RofSTO(2, 1.0242, r);
            });

            break;

        case 7: // Nitrogen

        	nMin = 1;
        	nMax = 2;
        	lMax = 1;

        	numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.354839 * RofSTO(1, 9.9051, r)
                + 0.472579 * RofSTO(1, 5.7429, r)
                - 0.001038 * RofSTO(3, 17.9816, r)
                + 0.208492 * RofSTO(2, 8.3087, r)
                + 0.001687 * RofSTO(2, 2.7611, r)
                + 0.000206 * RofSTO(2, 1.8223, r)
                + 0.000064 * RofSTO(2, 1.4191, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                - 0.067498 * RofSTO(1, 9.9051, r)
                + 0.434142 * RofSTO(1, 5.7429, r)
                - 0.000315 * RofSTO(3, 17.9816, r)
                - 0.080331 * RofSTO(2, 8.3087, r)
                - 0.374128 * RofSTO(2, 2.7611, r)
                - 0.522775 * RofSTO(2, 1.8223, r)
                - 0.207735 * RofSTO(2, 1.4191, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                + 0.006323 * RofSTO(2, 8.3490, r)
                + 0.082938 * RofSTO(2, 3.8827, r)
                + 0.260147 * RofSTO(2, 2.5920, r)
                + 0.418361 * RofSTO(2, 1.6946, r)
                + 0.308272 * RofSTO(2, 1.1914, r);
            });

            break;

        case 8: // Oxygen

            nMin = 1;
            nMax = 2;
            lMax = 1;
            numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.360063 * RofSTO(1, 11.2970, r)
                + 0.466625 * RofSTO(1, 6.5966, r)
                - 0.000918 * RofSTO(3, 20.5019, r)
                + 0.208441 * RofSTO(2, 9.5546, r)
                + 0.002018 * RofSTO(2, 3.2482, r)
                + 0.000216 * RofSTO(2, 2.1608, r)
                + 0.000133 * RofSTO(2, 1.6411, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                - 0.064363 * RofSTO(1, 11.2970, r)
                + 0.433186 * RofSTO(1, 6.5966, r)
                - 0.000275 * RofSTO(3, 20.5019, r)
                - 0.072497 * RofSTO(2, 9.5546, r)
                - 0.369900 * RofSTO(2, 3.2482, r)
                - 0.512627 * RofSTO(2, 2.1608, r)
                - 0.227421 * RofSTO(2, 1.6411, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                + 0.005626 * RofSTO(2, 9.6471, r)
                + 0.126618 * RofSTO(2, 4.3323, r)
                + 0.328966 * RofSTO(2, 2.7502, r)
                + 0.395422 * RofSTO(2, 1.7525, r)
                + 0.231788 * RofSTO(2, 1.2473, r);
            });

            break;

        case 9: // Fluorine

            nMin = 1;
            nMax = 2;
            lMax = 1;
            numOfOrbitals = (nMax*(nMax + 1))/2;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.377498 * RofSTO(1, 12.6074, r)
                + 0.443947 * RofSTO(1, 7.4101, r)
                - 0.000797 * RofSTO(3, 23.2475, r)
                + 0.213846 * RofSTO(2, 10.7416, r)
                + 0.002183 * RofSTO(2, 3.7543, r)
                + 0.000335 * RofSTO(2, 2.5009, r)
                + 0.000147 * RofSTO(2, 1.8577, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                - 0.058489 * RofSTO(1, 12.6074, r)
                + 0.426450 * RofSTO(1, 7.4101, r)
                - 0.000274 * RofSTO(3, 23.2475, r)
                - 0.063457 * RofSTO(2, 10.7416, r)
                - 0.358939 * RofSTO(2, 3.7543, r)
                - 0.516660 * RofSTO(2, 2.5009, r)
                - 0.239143 * RofSTO(2, 1.8577, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                + 0.004879 * RofSTO(2, 11.0134, r)
                + 0.130794 * RofSTO(2, 4.9962, r)
                + 0.337876 * RofSTO(2, 3.1540, r)
                + 0.396122 * RofSTO(2, 1.9722, r)
                + 0.225374 * RofSTO(2, 1.3632, r);
            });

            break;

        case 11: // Sodium

            nMin = 1;
            nMax = 3;
            lMax = 1;
            numOfOrbitals = 4;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.387167 * RofSTO(1, 15.3319, r)
                + 0.434278 * RofSTO(1, 9.0902, r)
                + 0.213027 * RofSTO(2, 13.2013, r)
                + 0.002205 * RofSTO(2, 4.7444, r)
                + 0.000627 * RofSTO(2, 3.1516, r)
                - 0.000044 * RofSTO(2, 2.4047, r)
                - 0.000649 * RofSTO(3, 28.4273, r)
                + 0.000026 * RofSTO(3, 1.3179, r)
                - 0.000023 * RofSTO(3, 0.8911, r)
                + 0.000008 * RofSTO(3, 0.6679, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.053722 * RofSTO(1, 15.3319, r)
                - 0.430794 * RofSTO(1, 9.0902, r)
                + 0.053654 * RofSTO(2, 13.2013, r)
                + 0.347971 * RofSTO(2, 4.7444, r)
                + 0.608890 * RofSTO(2, 3.1516, r)
                + 0.157462 * RofSTO(2, 2.4047, r)
                + 0.000280 * RofSTO(3, 28.4273, r)
                - 0.000492 * RofSTO(3, 1.3179, r)
                + 0.000457 * RofSTO(3, 0.8911, r)
                + 0.000016 * RofSTO(3, 0.6679, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                + 0.004308 * RofSTO(2, 13.6175, r)
                + 0.157824 * RofSTO(2, 6.2193, r)
                + 0.388545 * RofSTO(2, 3.8380, r)
                + 0.489339 * RofSTO(2, 2.3633, r)
                + 0.039759 * RofSTO(2, 1.5319, r);
            });

            // 3s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.011568 * RofSTO(1, 15.3319, r)
                - 0.072430 * RofSTO(1, 9.0902, r)
                + 0.011164 * RofSTO(2, 13.2013, r)
                + 0.057679 * RofSTO(2, 4.7444, r)
                + 0.089837 * RofSTO(2, 3.1516, r)
                + 0.042114 * RofSTO(2, 2.4047, r)
                - 0.000001 * RofSTO(3, 28.4273, r)
                - 0.182627 * RofSTO(3, 1.3179, r)
                - 0.471631 * RofSTO(3, 0.8911, r)
                - 0.408817 * RofSTO(3, 0.6679, r);
            });

            break;

        case 13: // Aluminium

            nMin = 1;
            nMax = 3;
            lMax = 1;
            numOfOrbitals = 5;
            bungeFunctions.reserve(numOfOrbitals);

            // 1s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.373865 * RofSTO(1, 18.1792, r)
                + 0.456146 * RofSTO(1, 10.8835, r)
                + 0.202560 * RofSTO(2, 15.7593, r)
                + 0.001901 * RofSTO(2, 5.7600, r)
                + 0.000823 * RofSTO(2, 4.0085, r)
                - 0.000267 * RofSTO(2, 2.8676, r)
                - 0.000560 * RofSTO(3, 33.5797, r)
                + 0.000083 * RofSTO(3, 2.1106, r)
                - 0.000044 * RofSTO(3, 1.3998, r)
                + 0.000013 * RofSTO(3, 1.0003, r);
            });

            // 2s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.061165 * RofSTO(1, 18.1792, r)
                - 0.460373 * RofSTO(1, 10.8835, r)
                + 0.055062 * RofSTO(2, 15.7593, r)
                + 0.297052 * RofSTO(2, 5.7600, r)
                + 0.750997 * RofSTO(2, 4.0085, r)
                + 0.064079 * RofSTO(2, 2.8676, r)
                + 0.000270 * RofSTO(3, 33.5797, r)
                - 0.001972 * RofSTO(3, 2.1106, r)
                + 0.000614 * RofSTO(3, 1.3998, r)
                - 0.000064 * RofSTO(3, 1.0003, r);
            });

            // 2p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                + 0.015480 * RofSTO(2, 14.4976, r)
                + 0.204774 * RofSTO(2, 6.6568, r)
                + 0.474317 * RofSTO(2, 4.2183, r)
                + 0.339646 * RofSTO(2, 3.0026, r)
                + 0.024290 * RofSTO(3, 11.0822, r)
                + 0.003529 * RofSTO(3, 1.6784, r)
                - 0.000204 * RofSTO(3, 1.0788, r)
                + 0.000199 * RofSTO(3, 0.7494, r);
            });

            // 3s RHF orbital
            bungeFunctions.push_back([&](double r){return 
                
                + 0.020024 * RofSTO(1, 18.1792, r)
                - 0.119051 * RofSTO(1, 10.8835, r)
                + 0.017451 * RofSTO(2, 15.7593, r)
                + 0.079185 * RofSTO(2, 5.7600, r)
                + 0.130917 * RofSTO(2, 4.0085, r)
                + 0.139113 * RofSTO(2, 2.8676, r)
                + 0.000038 * RofSTO(3, 33.5797, r)
                - 0.303750 * RofSTO(3, 2.1106, r)
                - 0.547941 * RofSTO(3, 1.3998, r)
                - 0.285949 * RofSTO(3, 1.0003, r);
            });

            // 3p RHF orbital 
            bungeFunctions.push_back([&](double r){return 
                
                - 0.001690 * RofSTO(2, 14.4976, r)
                - 0.048903 * RofSTO(2, 6.6568, r)
                - 0.058101 * RofSTO(2, 4.2183, r)
                - 0.090680 * RofSTO(2, 3.0026, r)
                - 0.001445 * RofSTO(3, 11.0822, r)
                + 0.234760 * RofSTO(3, 1.6784, r)
                + 0.496072 * RofSTO(3, 1.0788, r)
                + 0.359277 * RofSTO(3, 0.7494, r);
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

double AtomicOrbitalBasisManager::PseudoAtomicOrbitalvalue
		(const OrbitalQuantumNumbers& orbital, 
		 const dealii::Point<3>& evalPoint, 
		 const std::vector<double>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);

	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return RadialPseudoAtomicOrbital(n, l, r)
			* realSphericalHarmonics(l, m, theta, phi);
}
double AtomicOrbitalBasisManager::RadialPseudoAtomicOrbital(unsigned int n , unsigned int l, 
			 				         double  r)
{
   if( r >= rmax)
    return 0.0;
    if(r <= rmin)
        r = 0.01;
   double v = alglib::spline1dcalc(*radialSplineObject[n][l],r);
   // std::cout<<"$$$ "<<r<<"  "<<v<<std::endl;
    return v;
    
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

	return RadialPseudoAtomicOrbital(n, l, r)
			* realSphericalHarmonics(l, m, theta, phi);
}

double AtomicOrbitalBasisManager::PseudoAtomicOrbitalvalue
		(const OrbitalQuantumNumbers& orbital, 
		 const dealii::Point<3>& evalPoint, 
		 const std::array<double, 3>& atomPos){

	int n = orbital.n;
	int l = orbital.l;
	int m = orbital.m;

	double r{}, theta{}, phi{}; 

	auto relativeEvalPoint = relativeVector3d(evalPoint, atomPos);

	convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);

	return RadialPseudoAtomicOrbital(n, l, r)
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

//Function to create spline of PseudoAtomic Orbitals
void AtomicOrbitalBasisManager::CreatePseudoAtomicOrbitalBasis()
{
    if(PseudoAtomicOrbital == false)
        return;
    else
    {
        //std::cout<<"Entering CreatePseudoAtomicOrbitalBasis "<<std::endl;
        std::vector<std::vector<double>> values;
        std::string path = "../PAorbitals/PA_";
        for(int i = 0; i < n.size(); i++ )
        {
            if(m[i] == 0)
            {
                values.clear();
                std::string file = path + std::to_string(atomType)+"_"+std::to_string(n[i])+"_"+std::to_string(l[i])+".txt";
                //std::cout<<"Reading atomic orbital basis from file: "<<file<<std::endl;
                dftfe::dftUtils::readFile(2,values,file);
                int                 numRows = values.size();
                //std::cout<<"Number of Rows in "<<file<<" is"<<numRows<<std::endl;
                std::vector<double> xData(numRows), yData(numRows);
                for (int irow = 0; irow < numRows; ++irow)
                {
                    xData[irow] = values[irow][0];
                    yData[irow] = values[irow][1];
                    if (xData[irow] <= 0.00001)
                        yData[irow] = yData[irow+1]/xData[irow+1];
                    else
                        yData[irow] = yData[irow]/xData[irow];    
                } 
                rmax = xData[xData.size()-1];
                rmin = xData[1];
                yData[0] = yData[1];
                //std::cout<<"Value of the Datas at : "<<xData[0]<<" is "<<yData[0]<<std::endl;       
                alglib::real_1d_array x;
                x.setcontent(numRows, &xData[0]);
                alglib::real_1d_array y;
                y.setcontent(numRows, &yData[0]);
                alglib::ae_int_t             natural_bound_typeL = 0;
                alglib::ae_int_t             natural_bound_typeR = 1;
                alglib::spline1dinterpolant *spline = new alglib::spline1dinterpolant;
                alglib::spline1dbuildcubic(x,
                                 y,
                                 numRows,
                                 natural_bound_typeL,
                                 0.0,
                                 natural_bound_typeR,
                                 0.0,
                                 *spline);

                radialSplineObject[n[i]][l[i]] = spline; 

                double v = spline1dcalc(*radialSplineObject[n[i]][l[i]],0.5);         
                //std::cout<<" Value of spline at 0.5 is "<<v<<std::endl;
            }
        }

    }    
}