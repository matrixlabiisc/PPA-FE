

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <valarray>
#include "distributions.h"
#include "overlapPopulationAnalysis.h"
#include "matrixmatrixmul.h"


void
writeOrbitalDataIntoFile(const std::vector<std::vector<int>> &data,
                         const std::string &                  fileName)
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream outFile(fileName);
      if (outFile.is_open())
        {
          for (unsigned int irow = 0; irow < data.size(); ++irow)
            {
              for (unsigned int icol = 0; icol < data[irow].size(); ++icol)
                {
                  outFile << data[irow][icol];
                  if (icol < data[irow].size() - 1)
                    outFile << " ";
                }
              outFile << "\n";
            }

          outFile.close();
        }
    }
}



void
readBasisFile(const unsigned int             numColumns,
              std::vector<std::vector<int>> &data,
              const std::string &            fileName)
{
  std::vector<int> rowData(numColumns, 0.0);
  std::ifstream    readFile(fileName.c_str());
  if (readFile.fail())
    {
      std::cerr << "Error opening file: " << fileName.c_str() << std::endl;
      exit(-1);
    }

  //
  // String to store line and word
  //
  std::string readLine;
  std::string word;

  //
  // column index
  //
  int columnCount;

  if (readFile.is_open())
    {
      while (std::getline(readFile, readLine))
        {
          std::istringstream iss(readLine);

          columnCount = 0;

          while (iss >> word && columnCount < numColumns)
            rowData[columnCount++] = atoi(word.c_str());

          data.push_back(rowData);
        }
    }
  readFile.close();
}



// using this class we create an Array of Objects
// // each element corresponding to an atom type
// // corresponding atomPositions should be as an external array or suitable in
// a datastructure

void
constructQuantumNumbersHierarchy(unsigned int      n,
                                 unsigned int      l,
                                 std::vector<int> &rank)
{
  // assume the vector of size 0 has already been reserved with space for N
  // shells which is N(N+1)(2N+1)/6 orbitals for N shells N is maximum of the
  // principal quantum number over each atomType this function is called just
  // once for the whole program

  /*  	OrbitalQuantumNumbers orbitalTraverse;

for(unsigned int n = nstart; n <= nend; ++n) {
  for(unsigned int l = 0; l < n; ++l) {
    for(unsigned int tmp_m = 0; tmp_m <= 2*l; ++tmp_m) {

      orbitalTraverse.n = n;
      orbitalTraverse.l = l;
      orbitalTraverse.m = tmp_m - l;
      quantumNumHierarchy.push_back(orbitalTraverse);
    }
  }

}
*/
  rank.clear();
}

void
appendElemsOfRangeToFile(unsigned int start,
                         unsigned int end,
                         std::string  filename)
{
  std::ofstream outputFile;
  outputFile.open(filename, std::ofstream::out | std::ofstream::app);

  if (outputFile.is_open())
    {
      for (int i = start; i <= end; ++i)
        {
          outputFile << i << '\n';
        }
    }

  else
    {
      std::cerr << "Couldn't open " << filename << " file!!" << std::endl;
      exit(0);
    }

  outputFile.close();
  // it is usually not required to close the file
}

// this function assumes all filled Kohn-Sham orbitals/bands come first
// to find the first zero in the occupation number vector
unsigned int
numberOfFilledBands(const std::vector<double> &occupationNum)
{
  unsigned int numOfFilledKSorbitals =
    std::distance(std::begin(occupationNum),
                  std::find_if(std::begin(occupationNum),
                               std::end(occupationNum),
                               [](double x) { return (std::abs(x) < 1e-05); }));

  return numOfFilledKSorbitals;
}


spillFactors
spillFactorsOfProjection(const std::vector<double> &coeffMatrixVecOfProj,
                         const std::vector<double> &arrayVecOfProj,
                         const std::vector<double> &occupationNum)
{
  spillFactors spillvalues = {};

  unsigned int numOfFilledKSorbitals = numberOfFilledBands(occupationNum);
  unsigned int numOfKSOrbitals       = occupationNum.size();
  unsigned int totalDimOfBasis       = arrayVecOfProj.size() / numOfKSOrbitals;
  double       totalNumOfElectrons   = 0.0;

  spillvalues.projectabilities.resize(numOfKSOrbitals, 0.0);

  for (auto &n : occupationNum)
    {
      totalNumOfElectrons += n;
    }

  totalNumOfElectrons = 2 * totalNumOfElectrons;

  unsigned int index;

  std::valarray<double> spillForEachBand(1.0, numOfKSOrbitals);
  std::valarray<double> chargeSpillForEachBand(0.0, numOfFilledKSorbitals);

  for (size_t i = 0; i < numOfKSOrbitals; ++i)
    {
      for (size_t k = 0; k < totalDimOfBasis; ++k)
        {
          index = i + k * numOfKSOrbitals;

          spillvalues.projectabilities[i] +=
            coeffMatrixVecOfProj[index] * arrayVecOfProj[index];
        }

      spillForEachBand[i] -= spillvalues.projectabilities[i];

      if (i < numOfFilledKSorbitals)
        {
          chargeSpillForEachBand[i] =
            2 * spillForEachBand[i] * occupationNum[i];

          spillvalues.occupiedBandsSpilling += spillForEachBand[i];
          spillvalues.absOccupiedBandsSpilling += std::abs(spillForEachBand[i]);
        }
    }

  spillvalues.occupiedBandsSpilling /= numOfFilledKSorbitals;
  spillvalues.absOccupiedBandsSpilling /= numOfFilledKSorbitals;

  // actually we could have summed up everything in the above loop itself to
  // save space

  spillvalues.totalSpilling = spillForEachBand.sum() / numOfKSOrbitals;

  spillvalues.chargeSpilling = chargeSpillForEachBand.sum() /
                               (numOfFilledKSorbitals * totalNumOfElectrons);

  spillvalues.absTotalSpilling =
    std::abs(spillForEachBand).sum() / numOfKSOrbitals;

  spillvalues.absChargeSpilling = std::abs(chargeSpillForEachBand).sum() /
                                  (numOfFilledKSorbitals * totalNumOfElectrons);

  return spillvalues;
}


void
spillFactorsofProjectionwithCS(const std::vector<double> &C,
                               const std::vector<double> &Sold,
                               const std::vector<double> &occupationNum,
                               int                        m1,
                               int                        n1,
                               int                        m2,
                               int                        n2)
{
  int                 N = n1;
  std::vector<double> S(m2 * n2, 0.0);
  int                 count = 0;
  for (int i = 0; i < m2; i++)
    {
      for (int j = i; j < n2; j++)
        {
          S[i * n1 + j] = Sold[count];
          S[j * m1 + i] = Sold[count];
          count++;
        }
    }
  auto         temp                = matrixTmatrixmul(C, n1, m1, S, m2, n2);
  auto         O                   = matrixmatrixmul(temp, n1, n2, C, n1, m1);
  double       TSF                 = 0.0;
  double       CSF                 = 0.0;
  double       fCSF                = 0.0;
  double       TSFabs              = 0.0;
  double       CSFabs              = 0.0;
  double       fCSFabs             = 0.0;
  double       fsum                = 0.0;
  double       totalNumOfElectrons = 0.0;
  unsigned int numOfFilledKSorbitals =
    std::distance(std::begin(occupationNum),
                  std::find_if(std::begin(occupationNum),
                               std::end(occupationNum),
                               [](double x) { return (std::abs(x) < 1e-05); }));

  // unsigned int numOfKSOrbitals = occupationNum.size();

  for (int i = 0; i < N; i++)
    {
      TSF += 1 - O[i * N + i];
      TSFabs += std::fabs(1 - O[i * N + i]);

      if (i < numOfFilledKSorbitals)
        {
          CSF += (1 - O[i * N + i]);
          CSFabs += (std::fabs(1 - O[i * N + i]));
        }
      fCSF += (occupationNum[i] * O[i * N + i]);
      fCSFabs += std::fabs(occupationNum[i] * O[i * N + i]);
      fsum += occupationNum[i];
    }
  TSF /= N;
  TSFabs /= N;
  CSF /= (numOfFilledKSorbitals);
  CSFabs /= (numOfFilledKSorbitals);
  fCSF    = 1 - fCSF / fsum;
  fCSFabs = 1 - fCSFabs / fsum;
  std::cout << "Number of Filled KS orbitals: " << numOfFilledKSorbitals
            << std::endl;
  std::cout << "TSF: " << TSF << std::endl;
  std::cout << "TSFabs: " << TSF << std::endl;
  std::cout << "CSF: " << CSF << std::endl;
  std::cout << "CSFabs: " << CSFabs << std::endl;
  std::cout << "fCSF: " << fCSF << std::endl;
  std::cout << "fCSFabs: " << fCSFabs << std::endl;
}
void
spillFactorsofProjectionwithCS(const std::vector<double> &C_up,
                               const std::vector<double> &C_down,
                               const std::vector<double> &Sold,
                               const std::vector<double> &occupationNum,
                               int                        m1,
                               int                        n1,
                               int                        m2,
                               int                        n2)
{
  int                 N = n1;
  std::vector<double> S(m2 * n2, 0.0);
  int                 count = 0;
  for (int i = 0; i < m2; i++)
    {
      for (int j = i; j < n2; j++)
        {
          S[i * n1 + j] = Sold[count];
          S[j * m1 + i] = Sold[count];
          count++;
        }
    }
  auto         temp_up   = matrixTmatrixmul(C_up, n1, m1, S, m2, n2);
  auto         O_up      = matrixmatrixmul(temp_up, n1, n2, C_up, n1, m1);
  auto         temp_down = matrixTmatrixmul(C_down, n1, m1, S, m2, n2);
  auto         O_down    = matrixmatrixmul(temp_down, n1, n2, C_down, n1, m1);
  double       TSF_up    = 0.0;
  double       CSF_up    = 0.0;
  double       fCSF_up   = 0.0;
  double       TSF_down  = 0.0;
  double       CSF_down  = 0.0;
  double       fCSF_down = 0.0;
  double       fsum_up   = 0.0;
  double       fsum_down = 0.0;
  double       totalNumOfElectrons = 0.0;
  unsigned int numOfeigenValues    = occupationNum.size() / 2;
  unsigned int nstates_up          = 0;
  unsigned int nstates_down        = 0;
  for (int i = 0; i < N; i++)
    {
      TSF_up += (1 - O_up[i * N + i]);
      TSF_down += +(1 - O_down[i * N + i]);

      if (occupationNum[i] > 1e-05)
        {
          CSF_up += (1 - O_up[i * N + i]);
          nstates_up++;
        }
      if (occupationNum[i + numOfeigenValues] > 1e-05)
        {
          CSF_down += (1 - O_down[i * N + i]);

          nstates_down++;
        }

      fCSF_up += occupationNum[i] * O_up[i * N + i];
      fCSF_down += occupationNum[i + numOfeigenValues] * O_down[i * N + i];
      fsum_up += occupationNum[i];
      fsum_down += occupationNum[i + numOfeigenValues];
    }
  TSF_up /= N;
  TSF_down /= N;
  CSF_up /= nstates_up;
  CSF_down /= nstates_down;
  fCSF_up   = 1 - fCSF_up / fsum_up;
  fCSF_down = 1 - fCSF_down / fsum_down;
  std::cout << "Number of Filled KS orbitals fo spin up: " << nstates_up
            << std::endl;
  std::cout << "TSF: " << TSF_up << std::endl;
  std::cout << "CSF: " << CSF_up << std::endl;
  std::cout << "fCSF: " << fCSF_up << std::endl;
  std::cout << std::endl;
  std::cout << "Number of Filled KS orbitals fo spin down: " << nstates_down
            << std::endl;
  std::cout << "TSF: " << TSF_down << std::endl;
  std::cout << "CSF: " << CSF_down << std::endl;
  std::cout << "fCSF: " << fCSF_down << std::endl;
}



std::vector<double>
pCOOPvsEnergy(std::vector<double>        epsvalues,
              int                        globalBasisNum1,
              int                        globalBasisNum2,
              const std::vector<double> &SmatrixVec,
              const std::vector<double> &coeffArrayOfProj,
              const std::vector<double> &energyLevelsKS,
              const std::vector<int> &   occupationNum,
              std::vector<double> &      pCOOPcoeffs)
{
  int numOfKSOrbitals = energyLevelsKS.size();
  int numOfdatapoints = epsvalues.size();

  int totalDimOfBasis = coeffArrayOfProj.size() / numOfKSOrbitals;

  int minindex = std::min(globalBasisNum1, globalBasisNum2);
  int maxindex = std::max(globalBasisNum1, globalBasisNum2);

  int index =
    totalDimOfBasis * minindex - minindex * (minindex + 1) / 2 + maxindex;
  double S_ab = SmatrixVec[index];

  // std::cout << "Starting pCOOP vs energy calculation\n";

  std::vector<double> pCOOPvalues(numOfdatapoints, 0.0);

  // std::cout << "pCOOPvalues vector constructed\n";

  for (int j = 0; j < numOfKSOrbitals; ++j)
    {
      // std::cout << "entered the loop\n";
      // std::cout << "occupationNum = " << occupationNum[j] << '\n';

      if (occupationNum[j] == 0)
        {
          continue;
        } // { break; }

      // may be use continue instead of break
      // to see if there any further filled orbitals

      // std::cout << "Starting pCOOP vs energy calculation\n";

      pCOOPcoeffs[j] = coeffArrayOfProj[j + globalBasisNum1 * totalDimOfBasis] *
                       coeffArrayOfProj[j + globalBasisNum2 * totalDimOfBasis] *
                       S_ab * occupationNum[j];

      /* For testing H2 molecule case if everythng works fine

      std::cout << S_ab << '\n';
      std::cout << coeffArrayOfProj[j][globalBasisNum1] << '\n';
      std::cout << coeffArrayOfProj[j][globalBasisNum2] << '\n';
      std::cout << occupationNum[j] << '\n';

      std::cout << "pCOOPcoeff: " << tmpcoeff << '\n';

      */

      // std::cout << "Starting pCOOP vs energy calculation continues \n";

      for (int i = 0; i < numOfdatapoints; ++i)
        {
          pCOOPvalues[i] +=
            pCOOPcoeffs[j] * lorentzian(epsvalues[i], energyLevelsKS[j], 0.1);
        }

      std::cout << "pCOOP contribution of " << j
                << "th KSOrbital calculated!\n";
    }

  return pCOOPvalues;
}
// can use auto pCOOPvector to collect the return vector


// returns pCOHPvalues correspoding energy values vector
std::vector<double>
pCOHPvsEnergyTest(std::vector<double>        epsvalues,
                  int                        globalBasisNum1,
                  int                        globalBasisNum2, // a, b
                  const std::vector<double> &HmatrixVec,
                  const std::vector<double> &coeffArrayOfProj,
                  const std::vector<double> &energyLevelsKS,
                  const std::vector<int> &   occupationNum,
                  std::vector<double> &      pCOHPcoeffs)
{
  int numOfKSOrbitals = energyLevelsKS.size();
  int numOfdatapoints = epsvalues.size();

  int totalDimOfBasis = coeffArrayOfProj.size() / numOfKSOrbitals;

  int minindex = std::min(globalBasisNum1, globalBasisNum2);
  int maxindex = std::max(globalBasisNum1, globalBasisNum2);

  int index =
    totalDimOfBasis * minindex - minindex * (minindex + 1) / 2 + maxindex;
  double H_ab = HmatrixVec[index];

  // double tmpcoeff;

  // std::cout << "Starting pCOHP vs energy calculation\n";

  std::vector<double> pCOHPvalues(numOfdatapoints, 0.0);

  // std::cout << "pCOHPvalues vector constructed\n";

  for (int j = 0; j < numOfKSOrbitals; ++j)
    {
      // std::cout << "entered the loop\n";
      // std::cout << "occupationNum = " << occupationNum[j] << '\n';

      if (occupationNum[j] == 0)
        {
          continue;
        } // { break; }

      // may be use continue instead of break
      // to see if there any further filled orbitals

      // std::cout << "Starting pCOHP vs energy calculation\n";

      pCOHPcoeffs[j] = coeffArrayOfProj[j + globalBasisNum1 * totalDimOfBasis] *
                       coeffArrayOfProj[j + globalBasisNum2 * totalDimOfBasis] *
                       H_ab * occupationNum[j];

      /* For testing H2 molecule case if everythng works fine

      std::cout << H_ab << '\n';
      std::cout << coeffArrayOfProj[j][globalBasisNum1] << '\n';
      std::cout << coeffArrayOfProj[j][globalBasisNum2] << '\n';
      std::cout << occupationNum[j] << '\n';

      std::cout << "pCOHPcoeff: " << tmpcoeff << '\n';

      */

      // std::cout << "Starting pCOHP vs energy calculation continues \n";

      for (int i = 0; i < numOfdatapoints; ++i)
        {
          pCOHPvalues[i] +=
            pCOHPcoeffs[j] * lorentzian(epsvalues[i], energyLevelsKS[j], 0.1);
        }

      std::cout << "pCOHP contribution of " << j
                << "th KSOrbital calculated!\n";
    }

  return pCOHPvalues;
}
// can use auto pCOHPvector to collect the return vector
void
spillFactorsofProjectionwithCS(const std::vector<std::complex<double>> &C,
                               const std::vector<std::complex<double>> &Sold,
                               const std::vector<double> &occupationNum,
                               int                        m1,
                               int                        n1,
                               int                        m2,
                               int                        n2)
{
  int                 N = n1;
  std::vector<std::complex<double>> S(m2 * n2, std::complex<double> (0.0, 0.0));
  int                 count = 0;
  for (int i = 0; i < m2; i++)
    {
      for (int j = i; j < n2; j++)
        {
          S[i * n1 + j] = Sold[count];
          S[j * m1 + i] = Sold[count];
          count++;
        }
    }
  auto         temp                = matrixTmatrixmul(C, n1, m1, S, m2, n2);
  auto         O                   = matrixmatrixmul(temp, n1, n2, C, n1, m1);
  double       TSF                 = 0.0;
  double       CSF                 = 0.0;
  double       fCSF                = 0.0;
  double       TSFabs              = 0.0;
  double       CSFabs              = 0.0;
  double       fCSFabs             = 0.0;
  double       fsum                = 0.0;
  double       totalNumOfElectrons = 0.0;
  unsigned int numOfFilledKSorbitals =
    std::distance(std::begin(occupationNum),
                  std::find_if(std::begin(occupationNum),
                               std::end(occupationNum),
                               [](double x) { return (std::abs(x) < 1e-05); }));

  // unsigned int numOfKSOrbitals = occupationNum.size();

  for (int i = 0; i < N; i++)
    {
      TSF += 1 - O[i * N + i].real();
      TSFabs += std::fabs(1 - O[i * N + i].real());

      if (i < numOfFilledKSorbitals)
        {
          CSF += (1 - O[i * N + i].real());
          CSFabs += (std::fabs(1 - O[i * N + i].real()));
        }
      fCSF += (occupationNum[i] * O[i * N + i].real());
      fCSFabs += std::fabs(occupationNum[i] * O[i * N + i].real());
      fsum += occupationNum[i];
    }
  TSF /= N;
  TSFabs /= N;
  CSF /= (numOfFilledKSorbitals);
  CSFabs /= (numOfFilledKSorbitals);
  fCSF    = 1 - fCSF / fsum;
  fCSFabs = 1 - fCSFabs / fsum;
  std::cout << "Number of Filled KS orbitals: " << numOfFilledKSorbitals
            << std::endl;
  std::cout << "TSF: " << TSF << std::endl;
  std::cout << "TSFabs: " << TSF << std::endl;
  std::cout << "CSF: " << CSF << std::endl;
  std::cout << "CSFabs: " << CSFabs << std::endl;
  std::cout << "fCSF: " << fCSF << std::endl;
  std::cout << "fCSFabs: " << fCSFabs << std::endl;
}