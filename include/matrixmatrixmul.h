#pragma once
/*
 *
 *	The following functions are naive matrix matrix multiplications
 *	where the matrices are written as flattened vectors and the dimension
 *	is provided.
 *
 *	We have various variants matrix matrix^T, matrix matrix
 *	and matrix^T matrix and matrix vector multiplications,
 *	and for the matrix inverse we use the eigenlibrary
 *
 *	These would be eventually replaced by MKL or LAPACK or SCALAPACK
 *
 *
 */

#ifndef MATRIXMATRIXMUL_H_
#  define MATRIXMATRIXMUL_H_

#  include <vector>
#  include <iostream>
#  include <Eigen/Dense>
#  include <Eigen/Cholesky>

// only upper triangular matrix is provided as a vector
std::vector<double>
inverseOfOverlapMatrix(const std::vector<double> &SmatrixVec,
                       const size_t               matrixDim);

// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and B is m2 by n2 matrix
std::vector<double>
matrixmatrixmul(const std::vector<double> &,
                const unsigned int,
                const unsigned int,
                const std::vector<double> &,
                const unsigned int,
                const unsigned int);

// matrix A is m1 by n1 and B is m2 by n2
// this function achieves A^T * B, where both A and B are stored rowwise as a
// vector
std::vector<double>
matrixTmatrixmul(const std::vector<double> &A,
                 const unsigned int         m1,
                 const unsigned int         n1,
                 const std::vector<double> &B,
                 const unsigned int         m2,
                 const unsigned int         n2);


// matrix A is m1 by n1 and B is m2 by n2
// this function achieves A^T * B, where both A and B are stored rowwise as a
// vector this function is for cases where same matrix is transpose and
// multiplied in this case the result is a symmetric matrix so we store only
// upper triangular part
std::vector<double>
selfMatrixTmatrixmul(const std::vector<double> &A,
                     const unsigned int         m,
                     const unsigned int         n);

// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and BT is n2 by m2 matrix i.e. B is m2 by n2 matrix
// we pass B to the function which has been stored row wise
// or equivalently B is stored as a vector column wise
// and A*B is evaluated efficiently
std::vector<double>
matrixmatrixTmul(const std::vector<double> &A,
                 const unsigned int         m1,
                 const unsigned int         n1,
                 const std::vector<double> &B,
                 const unsigned int         m2,
                 const unsigned int         n2);



/**
 * Takes in the atomic orbital overlap matrix and the coefficient matrix.
 * Returns the coefficient matris that maps the othronormalised projected KS
 *wavefn. to orbital wave fn. Step1: Calculate O = C^{T}SC Step2: Eigen value
 *Decomposition of O = UDU^{T} Step3: Compute O^{-0.5} = UD^{-0.5}U^{T} Step4:
 *Cnew = CO^{-0.5}
 **/
std::vector<double>
OrthonormalizationofProjectedWavefn(const std::vector<double> &S,
                                    const unsigned int         m1,
                                    const unsigned int         n1,
                                    const std::vector<double> &C,
                                    const unsigned int         m2,
                                    const unsigned int         n2);

std::vector<double>
LowdenOrtho(const std::vector<double> &phi,
            int                        n_dofs,
            int                        N,
            const std::vector<double> &UpperS);

std::vector<double>
InvertPowerMatrix(double power, int N, const std::vector<double> &UpperS);

std::vector<double>
computeHprojOrbital(std::vector<double>              C,
                    int                              m,
                    int                              N,
                    std::vector<std::vector<double>> H);
#endif
