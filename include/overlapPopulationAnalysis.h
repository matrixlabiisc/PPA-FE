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
#  define OVERLAP_POPULATIONS_H_

#  include <iostream>
#  include <vector>
#  include <array>
#  include <algorithm>
#  include <numeric>
#  include <valarray>
#  include <fstream>
#  include <string>
#  include <cassert>
#  include "distributions.h"
#  include <dft.h>
#  include <dftUtils.h>
#  include <dftParameters.h>


template <typename T>
void
printVector(const std::vector<T> &vec)
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      for (const auto &elem : vec)
        {
          std::cout << elem << ' ';
        }
      std::cout << '\n';
    }
}

template <typename T, std::size_t N>
void
printArray(const std::array<T, N> &array)
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      for (const auto &elem : array)
        {
          std::cout << elem << ' ';
        }
      std::cout << '\n';
    }
}

template <typename T>
void
writeVectorToFile(const std::vector<T> &vec, std::string filename)
{
  std::ofstream outputFile(filename);

  if (outputFile.is_open())
    {
      for (const auto &elem : vec)
        {
          outputFile << std::setprecision(
                          std::numeric_limits<double>::max_digits10)
                     << elem << '\n';
        }

      outputFile.close();
    }

  else
    std::cout << "unable to create and open the" << filename << "file!\n";
}

template <typename T, std::size_t N>
void
writeArrayToFile(const std::array<T, N> &arr, std::string filename)
{
  std::ofstream outputFile(filename);

  if (outputFile.is_open())
    {
      for (const auto &elem : arr)
        {
          outputFile << elem << '\n';
        }

      outputFile.close();
    }

  else
    std::cout << "unable to create and open the" << filename << "file!\n";
}

template <typename T>
void
writeVectorAs2DMatrix(const std::vector<T> &vec,
                      unsigned int          numOfRows,
                      unsigned int          numOfColumns,
                      std::string           filename)
{
  assert(vec.size() == numOfRows * numOfColumns);

  std::ofstream outputFile(filename);

  unsigned int count = 0;

  if (outputFile.is_open())
    {
      for (unsigned int i = 0; i < numOfRows; ++i)
        {
          for (unsigned int j = 0; j < numOfColumns; ++j)
            {
              outputFile << vec[count] << " ";
              ++count;
            }

          outputFile << '\n';
        }

      outputFile.close();
    }

  // can also be implemented by a single for loop and if condition for next line

  else
    std::cout << "unable to create and open the" << filename << "file!\n";
}

struct spillFactors
{
  double absChargeSpilling;
  double absTotalSpilling;
  double absOccupiedBandsSpilling;

  double chargeSpilling;
  double totalSpilling;
  double occupiedBandsSpilling;

  std::vector<double> projectabilities;

  // note that total spilling in both variants is the spill factor for all bands
  // and charge spilling in both variants is the spill factor for occupied bands
  // later we can also check what projectability is and its relation to spill
  // factor occupiedBandsSpills average over only the occupied orbitals
};



// this function assumes all filled Kohn-Sham orbitals/bands come first
// to find the first zero in the occupation number vector
unsigned int
numberOfFilledBands(const std::vector<double> &occupationNum);


spillFactors
spillFactorsOfProjection(const std::vector<double> &coeffMatrixVecOfProj,
                         const std::vector<double> &arrayVecOfProj,
                         const std::vector<double> &occupationNum);

void
spillFactorsofProjectionwithCS(const std::vector<double> &C,
                               const std::vector<double> &Sold,
                               const std::vector<double> &occupationNum,
                               int                        m1,
                               int                        n1,
                               int                        m2,
                               int                        n2);
void
spillFactorsofProjectionwithCS(const std::vector<std::complex<double>> &C,
                               const std::vector<std::complex<double>> &Sold,
                               const std::vector<std::complex<double>> &occupationNum,
                               int                        m1,
                               int                        n1,
                               int                        m2,
                               int                        n2);                               
void
spillFactorsofProjectionwithCS(const std::vector<double> &C_up,
                               const std::vector<double> &C_down,
                               const std::vector<double> &Sold,
                               const std::vector<double> &occupationNum,
                               int                        m1,
                               int                        n1,
                               int                        m2,
                               int                        n2);


std::vector<double>
pCOOPvsEnergy(std::vector<double>        epsvalues,
              int                        globalBasisNum1,
              int                        globalBasisNum2,
              const std::vector<double> &SmatrixVec,
              const std::vector<double> &coeffArrayOfProj,
              const std::vector<double> &energyLevelsKS,
              const std::vector<int> &   occupationNum,
              std::vector<double> &      pCOOPcoeffs);


// returns pCOHPvalues correspoding energy values vector
std::vector<double>
pCOHPvsEnergyTest(std::vector<double>        epsvalues,
                  int                        globalBasisNum1,
                  int                        globalBasisNum2,
                  const std::vector<double> &HmatrixVec,
                  const std::vector<double> &coeffArrayOfProj,
                  const std::vector<double> &energyLevelsKS,
                  const std::vector<int> &   occupationNum,
                  std::vector<double> &      pCOHPcoeffs);



#endif
