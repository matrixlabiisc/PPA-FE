#pragma once
/*
*	This file contains the Bunge atomic orbitals
*	given using linear combinations of Slater type
*   orbitals for each atomic orbital, which have 
*   been used to solve Roothaan-Hartree-Fock equations 
*   giving as RHF orbitals 
*/

#ifndef BUNGE_ORBITALS_INFO_H_
#define BUNGE_ORBITALS_INFO_H_

#include <vector>
#include <array> 
#include <functional>
#include <cmath>
#include "mathUtils.h"

/** @brief Normalized Radial part of Slater Type Orbital 
 * 
 * 
 * 
 */
double ROfSTO(unsigned int n, double zetaEff, double r){

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
getRofBungeOrbitalBasisFuncs(unsigned int atomicNum){

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


#endif