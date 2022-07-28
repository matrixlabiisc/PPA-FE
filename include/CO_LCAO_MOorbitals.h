
#pragma once
/*
 *
 *	Molecular orbitals constructed from Extended Hueckel Theory
 * 	From the valence shell based Hydrogenic orbitals, here:
 * 	http://lampx.tugraz.at/~hadley/ss1/molecules/hueckel/CO/mo_CO.php
 * 	and http://lampx.tugraz.at/~hadley/ss1/molecules/hueckel/CO/CO.py
 *
 * 	also check:
 *	https://www.grandinetti.org/resources/Teaching/Chem4300/LectureCh18.pdf
 *
 */

#ifndef CO_LCAO_MOORBITALS_HPP_
#  define CO_LCAO_MOORBITALS_HPP_

#  include <functional>
#  include <vector>
#  include <array>
#  include <deal.II/grid/tria.h>
#  include "mathUtils.h"

// templates must be defined in header files

// in reality the above functions need not be in the
// header files (interface) as not used individually

double
MOCO1(const dealii::Point<3> &evalPoint);

double
MOCO2(const dealii::Point<3> &evalPoint);

double
MOCO3(const dealii::Point<3> &evalPoint);

double
MOCO4(const dealii::Point<3> &evalPoint);

double
MOCO5(const dealii::Point<3> &evalPoint);

double
MOCO6(const dealii::Point<3> &evalPoint);

double
MOCO7(const dealii::Point<3> &evalPoint);

double
MOCO8(const dealii::Point<3> &evalPoint);

std::vector<double>
assembleHamiltonianMatrixOfCO();

void
assembleCO_LCAO_MOorbitals(
  std::vector<double> &CO_MO_Energylevels,
  std::vector<std::function<double(const dealii::Point<3>)>> &MOsOfCO,
  std::vector<int> &                                          occupationNum);

#endif
