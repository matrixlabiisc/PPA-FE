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
#define MATRIXMATRIXMUL_H_

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky> 


// only upper triangular matrix is provided as a vector 
std::vector<double> 
inverseOfOverlapMatrix(const std::vector<double>& SmatrixVec, const size_t matrixDim);

// both matrices are full matrices written as rowwise flattened vectors 
// A is m1 by n1 matrix, and B is m2 by n2 matrix 
std::vector<double> 
matrixmatrixmul(const std::vector<double>&, unsigned int, unsigned int,
				const std::vector<double>&, unsigned int, unsigned int);

// matrix A is m1 by n1 and B is m2 by n2 
// this function achieves A^T * B, where both A and B are stored rowwise as a vector  
std::vector<double> 
matrixTmatrixmul(const std::vector<double>& A, unsigned int m1, unsigned int n1,
				 const std::vector<double>& B, unsigned int m2, unsigned int n2);


// matrix A is m1 by n1 and B is m2 by n2 
// this function achieves A^T * B, where both A and B are stored rowwise as a vector 
// this function is for cases where same matrix is transpose and multiplied
// in this case the result is a symmetric matrix so we store only upper triangular part 
std::vector<double> 
selfMatrixTmatrixmul(const std::vector<double>& A, unsigned int m, unsigned int n);

// both matrices are full matrices written as rowwise flattened vectors 
// A is m1 by n1 matrix, and BT is n2 by m2 matrix i.e. B is m2 by n2 matrix 
// we pass B to the function which has been stored row wise 
// or equivalently B is stored as a vector column wise
// and A*B is evaluated efficiently 
std::vector<double> 
matrixmatrixTmul(const std::vector<double>& A, unsigned int m1, unsigned int n1,
				 const std::vector<double>& B, unsigned int m2, unsigned int n2);



#endif