// header file for Gaussian and Lorentzian (also called cauchy distribution)
// both are normalized and approach Dirac delta when sigma and epsilon go to
// zero resp. https://www.quantstart.com/articles/Mathematical-Constants-in-C/
// https://en.cppreference.com/w/cpp/numeric/constants

// actually these functions are also available in Boost libraries
// check https://www.boost.org/doc/libs/1_76_0/libs/math/doc/html/dist.html
// for Normal (Gaussian) distribution and Cauchy-Lorentz distribution

#include "distributions.h"
#include <cmath>

double
gaussian(double x, double mu, double sigma = 0.1)
{
  return std::exp(-(x - mu) * (x - mu) / (2 * sigma * sigma)) /
         (sigma * std::sqrt(2 * M_PI));
}

double
lorentzian(double x, double mu, double epsilon = 0.1)
{
  return M_1_PI * epsilon / ((x - mu) * (x - mu) + epsilon * epsilon);
}
