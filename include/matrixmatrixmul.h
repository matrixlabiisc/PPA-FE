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
# include  <complex.h>


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

// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and B is m2 by n2 matrix
std::vector<std::complex<double>>
matrixmatrixmul(const std::vector<std::complex<double>> &,
                const unsigned int,
                const unsigned int,
                const std::vector<std::complex<double>> &,
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
// vector
std::vector<std::complex<double>>
matrixTmatrixmul(const std::vector<std::complex<double>> &A,
                 const unsigned int         m1,
                 const unsigned int         n1,
                 const std::vector<std::complex<double>> &B,
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


// matrix A is m1 by n1 and B is m2 by n2
// this function achieves A^T * B, where both A and B are stored rowwise as a
// vector this function is for cases where same matrix is transpose and
// multiplied in this case the result is a symmetric matrix so we store only
// upper triangular part
std::vector<std::complex<double>>
selfMatrixTmatrixmul(const std::vector<std::complex<double>> &A,
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


// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and BT is n2 by m2 matrix i.e. B is m2 by n2 matrix
// we pass B to the function which has been stored row wise
// or equivalently B is stored as a vector column wise
// and A*B is evaluated efficiently
std::vector<std::complex<double>>
matrixmatrixTmul(const std::vector<std::complex<double>> &A,
                 const unsigned int         m1,
                 const unsigned int         n1,
                 const std::vector<std::complex<double>> &B,
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



/**
 * Takes in the atomic orbital overlap matrix and the coefficient matrix.
 * Returns the coefficient matris that maps the othronormalised projected KS
 *wavefn. to orbital wave fn. Step1: Calculate O = C^{T}SC Step2: Eigen value
 *Decomposition of O = UDU^{T} Step3: Compute O^{-0.5} = UD^{-0.5}U^{T} Step4:
 *Cnew = CO^{-0.5}
 **/
std::vector<std::complex<double>>
OrthonormalizationofProjectedWavefn(const std::vector<std::complex<double>> &S,
                                    const unsigned int         m1,
                                    const unsigned int         n1,
                                    const std::vector<std::complex<double>> &C,
                                    const unsigned int         m2,
                                    const unsigned int         n2);

std::vector<double>
LowdenOrtho(const std::vector<double> &phi,
            int                        n_dofs,
            int                        N,
            const std::vector<double> &UpperS);


std::vector<std::complex<double>>
LowdenOrtho(const std::vector<std::complex<double>> &phi,
            int                        n_dofs,
            int                        N,
            const std::vector<std::complex<double>> &UpperS);            

std::vector<double>
InvertPowerMatrix(double power, int N, const std::vector<double> &UpperS);

std::vector<double>
computeHprojOrbital(std::vector<double>              C,
                    std::vector<double> &C_hat,
                    const unsigned int                              m,
                    const unsigned int                             N,
                    std::vector<double> &H);


std::vector<std::complex<double>>
computeHprojOrbital(std::vector<std::complex<double>>              C,
                    std::vector<std::complex<double>> &C_hat,
                    const unsigned int                              m,
                    const unsigned int                             N,
                    std::vector<double> &H);





std::vector<double> 
diagonalization(std::vector<double> S, int N, std::vector<double> &D);

std::vector<std::complex<double>> 
diagonalization(std::vector<std::complex<double>> &S, int N, std::vector<double> &D);

std::vector<double>
powerOfMatrix(double power, const std::vector<double> &D, const std::vector<double> &U,const unsigned int N, std::vector<double> Umod);

std::vector<std::complex<double>>
powerOfMatrix(double power, const std::vector<double> &D, const std::vector<std::complex<double>> &U,const unsigned int N, std::vector<std::complex<double>> Umod);

std::vector<double> 
TransposeMatrix(std::vector<double> &A, int N);

std::vector<std::complex<double>> 
TransposeMatrix(std::vector<std::complex<double>> &A, int N);

#endif
