// flattened array naive matrix manipulations 

#include "matrixmatrixmul.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky> 


std::vector<double> 
inverseOfOverlapMatrix(const std::vector<double>& SmatrixVec, const size_t matrixDim){

	std::vector<double> invS(matrixDim * matrixDim, 0.0); // storing full inverse matrix

	Eigen::MatrixXd S(matrixDim, matrixDim);

	unsigned int count = 0;

	for (unsigned int i = 0; i < matrixDim; ++i)
	{
		for (unsigned int j = i; j < matrixDim; ++j)
		{
			S(i, j) = SmatrixVec[count]; // since S was stored rowwise for upper triangular part 
			++count; 
		}
	}

	Eigen::MatrixXd UpperTriaS = S.selfadjointView<Eigen::Upper>(); // symmetric matrix 

	auto invS_direct = UpperTriaS.inverse(); // this works may be it uses cholesky at the back
	// surely uses LU as stated in the documentation 

	count = 0;

	for (unsigned int i = 0; i < matrixDim; ++i)
	{
		for (unsigned int j = 0; j < matrixDim; ++j)
		{
			// invS[count] = invS_eigen(i, j); // rowwise storage
			invS[count] = invS_direct(i, j);
			++count; 
		}
	}

	return invS;
}

// both matrices are full matrices written as rowwise flattened vectors 
// A is m1 by n1 matrix, and B is m2 by n2 matrix 
std::vector<double> 
matrixmatrixmul(const std::vector<double>& A, unsigned int m1, unsigned int n1,
				const std::vector<double>& B, unsigned int m2, unsigned int n2){

	assert((n1 == m2) && "Given matrices are not compatible for matrix multiplication");
	assert((A.size() == m1*n1) && "First matrix is not compatible with the specified dimensions");
	assert((B.size() == m2*n2) && "Second matrix is not compatible with the specified dimensions");

	unsigned int indexA, indexB, indexC = 0; 

	std::vector<double> C(m1*n2, 0.0); // initialized to zero 

	for (size_t i = 0; i < m1; ++i) // (i+1)th row of A
	{
		for (size_t j = 0; j < n2; ++j) // (j+1)th column of B
		{ 
			for (size_t k = 0; k < n1; ++k)
			{
				indexA = k + i*n1;
				indexB = k*n2 + j;

				C[indexC] += A[indexA] * B[indexB];
			}

			++indexC;
		}
	}

	return C;
}

// matrix A is m1 by n1 and B is m2 by n2 
// this function achieves A^T * B, where both A and B are stored rowwise as a vector  
std::vector<double> 
matrixTmatrixmul(const std::vector<double>& A, unsigned int m1, unsigned int n1,
				 const std::vector<double>& B, unsigned int m2, unsigned int n2){

	assert((m1 == m2) && "Given matrices are not compatible for matrix multiplication");
	assert((A.size() == m1*n1) && "First matrix is not compatible with the specified dimensions");
	assert((B.size() == m2*n2) && "Second matrix is not compatible with the specified dimensions");

	unsigned int indexA, indexB, indexC = 0; 

	std::vector<double> C(n1*n2, 0.0); // initialized to zero 

	// std::cout << "started matrixT matrix mul\n";

	for (size_t i = 0; i < n1; ++i) // (i+1)th column of A or row of A^T
	{
		for (size_t j = 0; j < n2; ++j) // (j+1)th column of B
		{ 

			for (size_t k = 0; k < m1; ++k)
			{
				indexA = i + k*n1;
				indexB = j + k*n2;

				C[indexC] += A[indexA] * B[indexB];
			}

			// std::cout << indexC << ' ';
			++indexC;
		}
	}

	// std::cout << '\n';
	return C;
}


// matrix A is m1 by n1 and B is m2 by n2 
// this function achieves A^T * B, where both A and B are stored rowwise as a vector 
// this function is for cases where same matrix is transpose and multiplied
// in this case the result is a symmetric matrix so we store only upper triangular part 
std::vector<double> 
selfMatrixTmatrixmul(const std::vector<double>& A, unsigned int m, unsigned int n){

	assert((A.size() == m*n) && "Given matrix is not compatible with the specified dimensions");

	unsigned int indexA, indexB, indexC = 0; 

	std::vector<double> C( (n*(n+1))/2, 0.0 ); // initialized to zero 

	// std::cout << "started matrixT matrix mul\n";

	for (size_t i = 0; i < n; ++i) // (i+1)th column of A or row of A^T
	{
		for (size_t j = i; j < n; ++j) // (j+1)th column of A
		{ 

			for (size_t k = 0; k < m; ++k)
			{
				indexA = i + k*n;
				indexB = j + k*n;

				C[indexC] += A[indexA] * A[indexB];
			}

			// std::cout << indexC << ' ';
			++indexC;
		}
	}

	// std::cout << '\n';
	return C;
}

// both matrices are full matrices written as rowwise flattened vectors 
// A is m1 by n1 matrix, and BT is n2 by m2 matrix i.e. B is m2 by n2 matrix 
// we pass B to the function which has been stored row wise 
// or equivalently B is stored as a vector column wise
// and A*B is evaluated efficiently 
std::vector<double> 
matrixmatrixTmul(const std::vector<double>& A, unsigned int m1, unsigned int n1,
				 const std::vector<double>& B, unsigned int m2, unsigned int n2){

	assert((n1 == n2) && "Given matrices are not compatible for matrix multiplication");
	assert((A.size() == m1*n1) && "First matrix is not compatible with the specified dimensions");
	assert((B.size() == m2*n2) && "Second matrix is not compatible with the specified dimensions");

	unsigned int indexA, indexBT, indexC = 0; 

	std::vector<double> C(m1*m2, 0.0); // initialized to zero 

	for (size_t i = 0; i < m1; ++i) // (i+1)th row of A
	{
		for (size_t j = 0; j < m2; ++j) // (j+1)th column of BT is (j+1)th row of B
		{ 
			indexA = i*n1;
			indexBT = j*n2;

			for (size_t k = 0; k < n1; ++k)
			{
				indexA = k + i*n1;
				indexBT = k + j*n2;

				C[indexC] += A[indexA] * B[indexBT];
			}

			++indexC;
		}
	}

	return C;
}
