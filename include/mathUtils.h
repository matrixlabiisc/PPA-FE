#pragma once
/*
 *
 *	Some math functions repeatedly used in the code
 *
 */

#ifndef MATH_UTILITIES_H_
#  define MATH_UTILITIES_H_

#  include <stdexcept>
#  include <vector>
#  include <array>
#  include <deal.II/grid/tria.h>

template <typename T>
inline T
pow2(T x)
{
  return x * x;
}

template <typename T>
inline T
pow3(T x)
{
  return x * x * x;
}

template <typename T>
inline T
pow4(T x)
{
  return x * x * x * x;
}

template <typename T>
inline T
pow5(T x)
{
  return x * x * x * x * x;
}

// for the above functions types are inferred

inline double
factorial(unsigned int N)
{
  if (N > 20)
    throw std::out_of_range(
      "This factorial function can handle 0 to 20 range only!!");

  double fac = 1.0;

  for (unsigned int i = 2; i <= N; ++i) // unsigned int i
    {
      fac *= i;
    }

  return fac;
}


inline double
distance3d(const dealii::Point<3> &   evalPoint,
           const std::vector<double> &atomPos)
{
  double sqdist = (evalPoint[0] - atomPos[0]) * (evalPoint[0] - atomPos[0]) +
                  (evalPoint[1] - atomPos[1]) * (evalPoint[1] - atomPos[1]) +
                  (evalPoint[2] - atomPos[2]) * (evalPoint[2] - atomPos[2]);

  return std::sqrt(sqdist);
}

inline double
distance3d(const dealii::Point<3> &     evalPoint,
           const std::array<double, 3> &atomPos)
{
  double sqdist = (evalPoint[0] - atomPos[0]) * (evalPoint[0] - atomPos[0]) +
                  (evalPoint[1] - atomPos[1]) * (evalPoint[1] - atomPos[1]) +
                  (evalPoint[2] - atomPos[2]) * (evalPoint[2] - atomPos[2]);

  return std::sqrt(sqdist);
}

inline double
norm3d(const std::vector<double> &x)
{
  return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

inline double
norm3d(const std::array<double, 3> &x)
{
  return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

inline std::vector<double>
relativeVector3d(const dealii::Point<3> &   evalPoint,
                 const std::vector<double> &atomPos)
{
  std::vector<double> relativeVec(3, 0.0);
  relativeVec[0] = evalPoint[0] - atomPos[0];
  relativeVec[1] = evalPoint[1] - atomPos[1];
  relativeVec[2] = evalPoint[2] - atomPos[2];

  return relativeVec;
}

inline std::array<double, 3>
relativeVector3d(const dealii::Point<3> &     evalPoint,
                 const std::array<double, 3> &atomPos)
{
  std::array<double, 3> relativeVec{};
  relativeVec[0] = evalPoint[0] - atomPos[0];
  relativeVec[1] = evalPoint[1] - atomPos[1];
  relativeVec[2] = evalPoint[2] - atomPos[2];

  return relativeVec;
}

#endif
