// header file for Gaussian and Lorentzian (also called cauchy distribution)
// both are normalized and approach Dirac delta when sigma and epsilon go to
// zero resp. https://www.quantstart.com/articles/Mathematical-Constants-in-C/
// https://en.cppreference.com/w/cpp/numeric/constants

// actually these functions are also available in Boost libraries
// check https://www.boost.org/doc/libs/1_76_0/libs/math/doc/html/dist.html
// for Normal (Gaussian) distribution and Cauchy-Lorentz distribution

#pragma once

#ifndef DISTRIBUTIONS_H_
#  define DISTRIBUTIONS_H_

#  include <cmath>

double
gaussian(double, double, double);

double
lorentzian(double, double, double);

#endif
