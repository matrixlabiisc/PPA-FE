// source file for CO LCAO Molecular orbitals

#include "CO_LCAO_MOorbitals.h"
#include <functional>
#include <vector>
#include <array>
#include <deal.II/grid/tria.h>
#include "mathUtils.h"

// check
// https://stackoverflow.com/questions/2268749/defining-global-constant-in-c

// the types in above are usually inferred, can also give default type T =
// double

// when is it advantageous to const pass by reference and just passing it?
// I guess it would depend on the cost of copying the data structure
// for fundamental datatypes this might not be a problem, but for
// derived (user-defined) data types and containers like std::vector and
// deallii::Point? std::vector might have efficient move semantics check:
// https://stackoverflow.com/questions/35032340/passing-containers-by-value-or-by-reference

// zeta is from the slater rules here

double
hydrogenic2sOrbital(double zeta, double r)
{
  return 0.25 * sqrt(pow3(zeta) / (2 * M_PI)) * (2 - zeta * r) *
         exp(-zeta * r / 2);
}

double
hydrogenic2pxOrbital(double zeta, double r, double xrel)
{
  return 0.25 * sqrt(pow5(zeta) / (2 * M_PI)) * xrel * exp(-zeta * r / 2);
}

double
hydrogenic2pyOrbital(double zeta, double r, double yrel)
{
  return 0.25 * sqrt(pow5(zeta) / (2 * M_PI)) * yrel * exp(-zeta * r / 2);
}

double
hydrogenic2pzOrbital(double zeta, double r, double zrel)
{
  return 0.25 * sqrt(pow5(zeta) / (2 * M_PI)) * zrel * exp(-zeta * r / 2);
}

double
MOCO1(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return -0.1389 * hydrogenic2sOrbital(zeta1, r1) +
         0.0080 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) +
         0.6492 * hydrogenic2sOrbital(zeta2, r2) +
         0.7478 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

double
MOCO2(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return +0.5243 * hydrogenic2sOrbital(zeta1, r1) -
         0.5884 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) -
         0.5660 * hydrogenic2sOrbital(zeta2, r2) -
         0.2420 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

double
MOCO3(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return +0.0000 * hydrogenic2sOrbital(zeta1, r1) +
         0.0000 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) -
         0.0939 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) +
         0.0939 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) +
         0.0000 * hydrogenic2sOrbital(zeta2, r2) +
         0.0000 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) -
         0.7008 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) +
         0.7008 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

double
MOCO4(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return +0.0000 * hydrogenic2sOrbital(zeta1, r1) +
         0.0000 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) +
         0.0939 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) +
         0.0939 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) +
         0.0000 * hydrogenic2sOrbital(zeta2, r2) +
         0.0000 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) +
         0.7008 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) +
         0.7008 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

double
MOCO5(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return -0.1707 * hydrogenic2sOrbital(zeta1, r1) +
         0.1532 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) -
         0.6456 * hydrogenic2sOrbital(zeta2, r2) +
         0.7285 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

double
MOCO6(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return +0.0000 * hydrogenic2sOrbital(zeta1, r1) +
         0.0000 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) -
         0.6615 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) -
         0.6615 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) +
         0.0000 * hydrogenic2sOrbital(zeta2, r2) +
         0.0000 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) +
         0.2499 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) +
         0.2499 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

double
MOCO7(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return +0.0000 * hydrogenic2sOrbital(zeta1, r1) +
         0.0000 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) -
         0.6615 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) +
         0.6615 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) +
         0.0000 * hydrogenic2sOrbital(zeta2, r2) +
         0.0000 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) +
         0.2499 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) -
         0.2499 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

double
MOCO8(const dealii::Point<3> &evalPoint)
{
  const std::array<double, 3> atomPos1{-1.06580553409, 0.0, 0.0}; // Carbon
  const std::array<double, 3> atomPos2{+1.06580553409, 0.0, 0.0}; // Oxygen

  auto   relVec1 = relativeVector3d(evalPoint, atomPos1);
  double r1      = norm3d(relVec1);

  auto   relVec2 = relativeVector3d(evalPoint, atomPos2);
  double r2      = norm3d(relVec2);

  const double zeta1 =
    3.25; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Carbon
  const double zeta2 =
    4.55; // Slater-rule zeta for 2s, 2px, 2py, 2pz for Oxygen

  return -0.7571 * hydrogenic2sOrbital(zeta1, r1) -
         0.6462 * hydrogenic2pxOrbital(zeta1, r1, relVec1[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta1, r1, relVec1[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta1, r1, relVec1[2]) -
         0.0089 * hydrogenic2sOrbital(zeta2, r2) -
         0.0950 * hydrogenic2pxOrbital(zeta2, r2, relVec2[0]) +
         0.0000 * hydrogenic2pyOrbital(zeta2, r2, relVec2[1]) +
         0.0000 * hydrogenic2pzOrbital(zeta2, r2, relVec2[2]);
}

// we assume the Hamiltonian matrix has declared as std::vector<double>
// usually vector of vector may not be a good practice
// so we are storing the H matrix as single vector row-wise
// remember Hamiltonian is a square matrix and self-adjoint
// and in case of real wave functions it is symmetric

// would it be better to take it as an Eigen matrix?

// For now we are going to do pCOHP specific to Hydrogenic orbitals
// so there is no need to project the Hamiltonian matrix on to a
// new basis like STOs to perform the pCOHP calculations
// so in this case we just have to replace S with H in the pCOOP
// calculations and similarly store all pCOHP coefficients and
// and also the pCOOPvsEnergy plots

std::vector<double>
assembleHamiltonianMatrixOfCO()
{
  std::vector<double> UpperTriaHvec{

    -90.63,  20.73, 0,      0,      -52.54,  -32.39, 0, 0, -102.59, 0, 0,
    59.35,   43.97, 0,      0,      -87.01,  0,      0, 0, -31.20,  0, -87.01,
    0,       0,     0,      -31.20, -111.57, -12.40, 0, 0, -116.66, 0, 0,
    -109.27, 0,     -109.27

  };

  return UpperTriaHvec;
}

void
assembleCO_LCAO_MOorbitals(
  std::vector<double> &CO_MO_Energylevels,
  std::vector<std::function<double(const dealii::Point<3>)>> &MOsOfCO,
  std::vector<int> &                                          occupationNum)
{
  int numOfMOorbitals = 8;

  CO_MO_Energylevels.reserve(numOfMOorbitals);
  MOsOfCO.reserve(numOfMOorbitals);
  occupationNum.reserve(numOfMOorbitals);

  double energyOfHOMO = -101.26; // to be taken as the reference energy, Zero!

  // vertical dotted line required here in the plot may be


  // Filled in increasing order of energy (in electron Volt (eV) units)

  // 1st MO of CO

  CO_MO_Energylevels.push_back(-127.43 - energyOfHOMO);
  MOsOfCO.push_back(MOCO1);
  occupationNum.push_back(2);


  // 2nd MO of CO

  CO_MO_Energylevels.push_back(-112.52 - energyOfHOMO);
  MOsOfCO.push_back(MOCO2);
  occupationNum.push_back(2);

  // 3rd MO of CO

  CO_MO_Energylevels.push_back(-109.68 - energyOfHOMO);
  MOsOfCO.push_back(MOCO3);
  occupationNum.push_back(2);


  // 4th MO of CO

  CO_MO_Energylevels.push_back(-109.68 - energyOfHOMO);
  MOsOfCO.push_back(MOCO4);
  occupationNum.push_back(2);


  // 5th MO of CO

  CO_MO_Energylevels.push_back(-101.26 - energyOfHOMO);
  MOsOfCO.push_back(MOCO5);
  occupationNum.push_back(2);


  // 6th MO of CO

  CO_MO_Energylevels.push_back(-83.31 - energyOfHOMO);
  MOsOfCO.push_back(MOCO6);
  occupationNum.push_back(0);


  // 7th MO of CO

  CO_MO_Energylevels.push_back(-83.31 - energyOfHOMO);
  MOsOfCO.push_back(MOCO7);
  occupationNum.push_back(0);

  // 8th MO of CO

  CO_MO_Energylevels.push_back(-74.79 - energyOfHOMO);
  MOsOfCO.push_back(MOCO8);
  occupationNum.push_back(0);

  return;
}
