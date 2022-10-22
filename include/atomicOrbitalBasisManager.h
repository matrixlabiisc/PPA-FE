#pragma once

#ifndef ATOMIC_ORBITAL_BASIS_MANAGER_H_
#  define ATOMIC_ORBITAL_BASIS_MANAGER_H_

#  include <boost/math/special_functions/spherical_harmonic.hpp>
#  include <boost/math/special_functions/laguerre.hpp>

#  include <vector>
#  include <array>
#  include <cmath>
#  include <fstream>
#  include <iostream>
#  include <sstream>
#  include <functional>
#  include <fileReaders.h>
#  include <dftParameters.h>
#  include <dftUtils.h>
#  include <interpolation.h>
#  include <headers.h>
#  include <deal.II/grid/tria.h>

#  include "mathUtils.h"
#  include "matrixmatrixmul.h"



struct OrbitalQuantumNumbers
{
  unsigned int n;
  unsigned int l;
  int          m;
};

struct LocalAtomicBasisInfo
{
  unsigned int atomID;     // required for atomPos
  unsigned int atomTypeID; // required for atomBasis
  unsigned int n;          // for getting the prinicipal QN
  unsigned int l;          // for getting the angular momoemtum QN
  int          m;          // for getting the magnetic QN
};

class AtomicOrbitalBasisManager // would be instantiated for each atom type
{
private:
  unsigned int basisDataForm;

  // !(Analytical form = 1) or generatorFunc >= 1, Tabular data = 0
  // generatorFunc = 1 Slater Type Orbitals, = 2 can be used for Hydrogenic
  // orbitals = 3 can be used for Bunge orbitals

  bool normalizedOrNot;

  // So that we can perform normalization if required

  // the below two can be external inputs
  // unsigned int numOfsuchAtoms; // eg, how many number of Hydrogen atoms

  unsigned int dimOfBasis;

  // atomic basis hierarchy for this atom type
  // if different atoms of same atoms type are chosen to have different basis
  // dim: set the above the maximum of all instead of the above n can also be
  // mentioned if above is mentioned may be it wouldn't fit the complete n, l
  // and m hierarchy would be better to have the nearest complete shell of
  // orbitals



  // the effective charge? = 0 implies no data provided try other basis

  unsigned int numOfSplineFuncParams; // this depends on the type of basis used

  std::vector<double> splineInterpolationConstants;

  // orbitnumsvector.pushback(atomhierarch(n,l));
  // a place for parameters is necessary
  // usually atomic basis hierarchy is through n, l, and m and decided wholely
  // by n the radial part is only dependent on n for a given atom basis the
  // angular part Y_{l,m} remains same for any basis parameter family

  // may be function pointer would be a big mess better make the
  // basisGeneratorFuncs members std::function<double(unsigned int, double,
  // double, double)> basisGeneratorFunc; the only problem with the above form
  // is this is the 'int' doesn't make sense for Analytical form basis functions
  // which we wish to assemble as std::vector of funcptrs usually unique
  // analytical forms for each atomic orbital are way of approach but for
  // testing purpose we can externally assemble a vector of functptrs to test
  // the integration and overlapmatrix calculations for validation purposes


  // the function for overlap matrix and projection operator are external
  // functions do not need to be friend functions as the above
  // basisGeneratorFunc can access all members and the member functions

  std::vector<std::function<double(double)>> ROfBungeBasisFunctions;


public:
  // constructor with initialization list

  AtomicOrbitalBasisManager(unsigned int atype, unsigned int btype, bool nor)
    : atomType(atype)
    , basisDataForm(btype)
    , normalizedOrNot(nor)
  {
    if (basisDataForm == 0)
      {
        PseudoAtomicOrbital = true;
      }


    else if (basisDataForm == 1) // Bunge orbitals
      {
        getRofBungeOrbitalBasisFuncs(atomType);
      }
  }
  unsigned int atomType; // Atomic number, isotope cases can be considered later
  std::vector<int> n;
  std::vector<int> l;
  std::vector<int> m;
  double           rmax, rmin;
  bool             PseudoAtomicOrbital = false;
  double maxRadialcutoff = -1.0;
  void
  CreatePseudoAtomicOrbitalBasis();
  std::map<unsigned int, std::map<unsigned int, alglib::spline1dinterpolant *>>
    radialSplineObject;
  std::map<unsigned int, std::map<unsigned int, std::function<double(double)>>>
         ROfBungeBasisFunct;
  double zeta;
  void
  setorbitalnums();
  int
  sizeofbasis()
  {
    return (m.size());
  }

  // check various types of format
  // https://www.basissetexchange.org/
  // https://github.com/MolSSI-BSE/basis_set_exchange

  void
  splineInterpolateBasisData()
  {}

  double
  splineInterpolationFunc(unsigned int, double, double);

  double
  RofSTO(unsigned int n, double zetaEff, double r);

  double
  RofHydrogenicOrbital(unsigned int n,
                       unsigned int l,
                       double       zetaEff,
                       double       r);

  void
  getRofBungeOrbitalBasisFuncs(unsigned int atomicNum);

  double
  radialPartofSlaterTypeOrbital(unsigned int n, double r);

  // in general radial part might depend on l as well, but not in the case of
  // STO

  double
  radialPartOfHydrogenicOrbital(unsigned int n, unsigned int l, double r);

  double
  radialPartOfBungeOrbital(unsigned int n, unsigned int l, double r);


  double
  realSphericalHarmonics(unsigned int l, short int m, double theta, double phi);

  double
  slaterTypeOrbital(const OrbitalQuantumNumbers &orbital,
                    const dealii::Point<3> &     evalPoint,
                    const std::vector<double> &  atomPos);


  double
  RadialPseudoAtomicOrbital(unsigned int n, unsigned int l, double r);



  double
  slaterTypeOrbital(const OrbitalQuantumNumbers &orbital,
                    const dealii::Point<3> &     evalPoint,
                    const std::array<double, 3> &atomPos);

  double
  hydrogenicOrbital(const OrbitalQuantumNumbers &orbital,
                    const dealii::Point<3> &     evalPoint,
                    const std::vector<double> &  atomPos);

  double
  hydrogenicOrbital(const OrbitalQuantumNumbers &orbital,
                    const dealii::Point<3> &     evalPoint,
                    const std::array<double, 3> &atomPos);

  double
  bungeOrbital(const OrbitalQuantumNumbers &orbital,
               const dealii::Point<3> &     evalPoint,
               const std::vector<double> &  atomPos);

  double
  bungeOrbital(const OrbitalQuantumNumbers &orbital,
               const dealii::Point<3> &     evalPoint,
               const std::array<double, 3> &atomPos);

  double
  PseudoAtomicOrbitalvalue(const OrbitalQuantumNumbers &orbital,
                           const dealii::Point<3> &     evalPoint,
                           const std::vector<double> &  atomPos);
    double
  PseudoAtomicOrbitalvalue(const OrbitalQuantumNumbers &orbital,
                           const dealii::Point<3> &     evalPoint,
                           const std::vector<double> &  atomPos, double r, double theta, double phi);                         
  double
  PseudoAtomicOrbitalvalue(const OrbitalQuantumNumbers &orbital,
                           const dealii::Point<3> &     evalPoint,
                           const std::array<double, 3> &atomPos);
  ~AtomicOrbitalBasisManager()
  {}
};

// using this class we create an Array of Objects
// each element corresponding to an atom type
// corresponding atomPositions should be as an external array or suitable in a
// datastructure

inline void
convertCartesianToSpherical(
  const std::vector<double> &x, // relative pos vec wrt to atomPos
  double &                   r,
  double &                   theta,
  double &                   phi)
{
  double tolerance = 1e-12;

  r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

  // if (std::fabs(r - 0.0) <= tolerance) // why fabs ? and r - 0.0 ?

  if (r < tolerance)
    {
      theta = 0.0;
      phi   = 0.0;
    }

  else
    {
      theta = std::acos(x[2] / r);

      // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
      // If yes, assign phi = 0.0.
      // NOTE: In case theta = 0 or PI, phi is undetermined. The actual
      // value of phi doesn't matter in computing the enriched function
      // value or its gradient. We assign phi = 0.0 here just as a dummy
      // value

      if ((std::fabs(theta - 0.0) >= tolerance) &&
          (std::fabs(theta - M_PI) >= tolerance))
        phi = std::atan2(x[1], x[0]);
      else
        phi = 0.0;
    }
}

inline void
convertCartesianToSpherical(
  const std::array<double, 3> &x, // relative pos vec wrt to atomPos
  double &                     r,
  double &                     theta,
  double &                     phi)
{
  double tolerance = 1e-12;

  r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

  // if (std::fabs(r - 0.0) <= tolerance) // why fabs ? and r - 0.0 ?

  if (r < tolerance)
    {
      theta = 0.0;
      phi   = 0.0;
    }

  else
    {
      theta = std::acos(x[2] / r);

      // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
      // If yes, assign phi = 0.0.
      // NOTE: In case theta = 0 or PI, phi is undetermined. The actual
      // value of phi doesn't matter in computing the enriched function
      // value or its gradient. We assign phi = 0.0 here just as a dummy
      // value

      if ((std::fabs(theta - 0.0) >= tolerance) &&
          (std::fabs(theta - M_PI) >= tolerance))
        phi = std::atan2(x[1], x[0]);
      else
        phi = 0.0;
    }
}

inline unsigned int
numofOrbitalsUntilShell(unsigned int n)
{
  return (n * (n + 1) * (2 * n + 1)) / 6;
}


inline unsigned int
numOfOrbitalsForShellCount(unsigned int n1, unsigned int n2)
{
  unsigned int numOrbitals = 0;

  if (n1 == 1)
    {
      return (n2 * (n2 + 1) * (2 * n2 + 1)) / 6;
    }

  for (unsigned int n = n1; n <= n2; ++n)
    numOrbitals += n * n;

  return numOrbitals;
}
// if n1 = 1 and n2 = n then numOrbitals = n(n+1)(2n+1)/6

inline unsigned int
numOfOrbitalsForShellCount(unsigned int n)
{
  return n * n;
}

double
hydrogenMoleculeBondingOrbital(const dealii::Point<3> &evalPoint);

#endif
