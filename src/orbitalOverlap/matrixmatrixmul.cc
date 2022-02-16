// flattened array naive matrix manipulations 

#include "matrixmatrixmul.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky> 
#include <mkl.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <vector>
#include <math.h>
#include<iostream>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>



std::vector<double> 
inverseOfOverlapMatrix(const std::vector<double>& SmatrixVec, const size_t matrixDim){

	//std::vector<double> invS(matrixDim * matrixDim, 0.0); // storing full inverse matrix
	std::vector<double> S(matrixDim * matrixDim, 0.0);
	int count = 0;
	for(int i = 0; i < matrixDim; i++)
	{
		for(int j=0; j < matrixDim; j++)
		{
			if(j >=i)
			{
				S[i*matrixDim+j] = SmatrixVec[count];
				count++;
			}
		}
	}

	
	  int                info;
	  const unsigned int N = matrixDim;
      const unsigned int lwork = 1 + 6*N +
                                 2 * N*N,
                         liwork = 3 + 5 *N;
      std::vector<int>    iwork(liwork, 0);
      const char          jobz = 'V', uplo = 'L';
      std::vector<double> work(lwork);
	  std::vector<double> D(N,0.0);
      dftfe::dsyevd_(&jobz,
              &uplo,
              &N,
              &S[0],
              &N,
              &D[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
    work.clear();
    iwork.clear();
    std::vector<double>().swap(work);
    std::vector<int>().swap(iwork);
	if(info > 0)
		std::cout<<"Eigen Value Decomposition Failed!!"<<std::endl;
	if(dftfe::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	{
		std::cout<<"EigenValues of S are: "<<std::endl;
		double min = 999999;
		for(int i = 0; i < N; i++)
		{	
			std::cout<<D[i]<<" ";
			if(min > D[i])
				min = D[i];
		}		
		std::cout<<std::endl;
		std::cout<<"Min Eigenvalue of S is: "<<min<<std::endl;
	}		



	std::vector<double>D_inv(N*N,0.0);
	std::vector<double> Uvector(N*N,0);
	for(int i = 0; i <N; i++)
		D_inv[i*N+i] = pow(D[i],-1.0);

	for(int i =0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			Uvector[i*N+j] = S[i*N+j];
	}	
	//U is in column major form.. O = U'DU
    auto D_invU = matrixmatrixmul(D_inv,N,N,Uvector,N,N);
	auto invS = matrixTmatrixmul(Uvector,N,N,D_invU,N,N);

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


std::vector<double>
OrthonormalizationofProjectedWavefn(const std::vector<double> &Sold, unsigned int m1, unsigned int n1,
									const std::vector<double> &C, unsigned int m2, unsigned int n2
									)
{
	//m1: Basis Dimension
	//n1: Total number of Atomic Orbitals
	//m2: Total Number of Atomic Orbitals
	//n2: Number of KS orbitals.
	assert((m2 == n1) && "Number of Atomic Orbitals not consistent");
	
	
	//std::cout<<"#Begin OrthoNormalization"<<std::endl;
	std::vector<double> S(m1*n1,0.0);
	int count = 0;
	for(int i = 0; i < m1; i++)
	{
		for(int j = i; j < n1; j++)
		{
			S[i*n1+j] = Sold[count];
			S[j*m1+i] = Sold[count];
			count++;
		}
	}
	/*
	for (int i = 0; i < m1; i++)
	{
		for(int j = 0; j < n1; j++)
			std::cout<<S[i*n1+j]<<" ";
		std::cout<<std::endl;
	}


	std::cout<<"#Created S full"<<std::endl;
	std::cout<<"matrix C is: "<<std::endl;
		for (int i = 0; i < m2; i++)
	{
		for(int j = 0; j < n2; j++)
			std::cout<<C[i*n1+j]<<" ";
		std::cout<<std::endl;
	}
	*/
	//std::cout<<"#B=CtS"<<std::endl;
	auto B = matrixTmatrixmul(C,m2,n2,S,m1,n1);
	/*	for (int i = 0; i < n2; i++)
	{
		for(int j = 0; j < n1; j++)
			std::cout<<B[i*n1+j]<<" ";
		std::cout<<std::endl;
	} 
	std::cout<<"#O=(CtS)C"<<std::endl; */
	auto O = matrixmatrixmul(B,n2,n1,C,m2,n2);
	/*	for (int i = 0; i < n2; i++)
	{
		for(int j = 0; j < n2; j++)
			std::cout<<O[i*n2+j]<<" ";
		std::cout<<std::endl;
	} */	
	int N = n2;
	//double D[N];
	std::vector<double> D(N,0.0);
	count = 0;
	//double upperO[N*N];
	std::vector<double> upperO(N*N,0.0);
	// std::cout<<"#Begin Upper Triangle creation"<<std::endl;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			if(j>=i)
			upperO[i*N+j] = O[i*N+j];
			else
			upperO[i*N+j] = 0.0;
			
		} 
	}/*
	for (int i = 0; i < n2; i++)
	{
		for(int j = 0; j < n2; j++)
			std::cout<<upperO[i*n2+j]<<" ";
		std::cout<<std::endl;
	}
	std::cout<<"#Completed Upper Triangle creation"<<std::endl;	

      std::cout<<"#Begin Eigen Value Decomposition"<<std::endl; */
	  int                info;
      const  int lwork = 1 + 6*N +
                                 2 * N*N,
                         liwork = 3 + 5 *N;
      std::vector<int>    iwork(liwork, 0);
      const char          jobz = 'V', uplo = 'L';
      std::vector<double> work(lwork);

      dsyevd_(&jobz,
              &uplo,
              &N,
              &upperO[0],
              &N,
              &D[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
    work.clear();
    iwork.clear();
    std::vector<double>().swap(work);
    std::vector<int>().swap(iwork);
	if(info > 0)
		std::cout<<"Eigen Value Decomposition Falied!!"<<std::endl;
	/*else
	{
		std::cout<<"The diagonal entried are:"<<std::endl;
		for(int i = 0; i < N; i++)
			std::cout<<D[i]<<" ";
		std::cout<<std::endl;
		std::cout<<"The U matrix is: "<<std::endl;
		for(int i = 0; i < N; i++)
		{
			for(int j=0; j < N; j++)
				std::cout<<upperO[i*N+j]<<" ";
			std::cout<<std::endl;	
		}	
	} */
	std::vector<double>D_half(N*N,0.0);
	std::vector<double> upperOvector(N*N,0);
	for(int i = 0; i <N; i++)
		D_half[i*N+i] = pow(D[i],-0.5);
	/*std::cout<<"D half matrix: "<<std::endl;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
			std::cout<<D_half[i*N+j]<<" ";
		std::cout<<std::endl; 	
	} */
	for(int i =0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			upperOvector[i*N+j] = upperO[i*N+j];
	}	
	//U is in column major form.. O = U'DU
    auto D_hfU = matrixmatrixmul(D_half,N,N,upperOvector,N,N);
	auto O_half = matrixTmatrixmul(upperOvector,N,N,D_hfU,N,N);

	auto Cnew = matrixmatrixmul(C,m2,n2,O_half,n2,n2);

	auto temp = matrixTmatrixmul(Cnew,n2,n2,S,m1,n1);
	auto temp2 = matrixmatrixmul(temp,m2,n1,Cnew,m2,n2);
/*	std::cout<<"The I matrix is: "<<std::endl;
	for(int i = 0; i < m2; i++)
	{
		for(int j = 0; j <n2; j++)
			std::cout<<temp2[i*n2+j]<<" ";
		std::cout<<std::endl;
	}
	*/
	
	
	return Cnew;



}

std::vector<double>  LowdenOrtho(const std::vector<double> &phi, int n_dofs, int N, const std::vector<double> &UpperS)
{
	std::vector<double> S(N*N,0.0);
	int count = 0;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			S[i*N+j] = 0.0;
			if(j >= i)
			{
				S[i*N+j]=UpperS[count]; 
				count++;
			}
		}
	}
      //std::cout<<"#Begin Eigen Value Decomposition"<<std::endl;
	  int                info;
      const  int lwork = 1 + 6*N +
                                 2 * N*N,
                         liwork = 3 + 5 *N;
      std::vector<int>    iwork(liwork, 0);
      const char          jobz = 'V', uplo = 'L';
      std::vector<double> work(lwork);
	  std::vector<double> D(N,0.0);	
      dsyevd_(&jobz,
              &uplo,
              &N,
              &S[0],
              &N,
              &D[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
    work.clear();
    iwork.clear();
    std::vector<double>().swap(work);
    std::vector<int>().swap(iwork);
	if(info > 0)
		std::cout<<"Eigen Value Decomposition Falied!!"<<std::endl;
/*	else
	{
		std::cout<<"The diagonal entried are:"<<std::endl;
		for(int i = 0; i < N; i++)
			std::cout<<D[i]<<" ";
		std::cout<<std::endl;
		std::cout<<"The U matrix is: "<<std::endl;
		for(int i = 0; i < N; i++)
		{
			for(int j=0; j < N; j++)
				std::cout<<S[j*N+i]<<" ";
			std::cout<<std::endl;	
		}	
	} */
	std::vector<double>D_half(N*N,0.0);
	std::vector<double> Uvector(N*N,0);
	for(int i = 0; i <N; i++)
		D_half[i*N+i] = pow(D[i],-0.5);
	/*std::cout<<"D half matrix: "<<std::endl;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
			std::cout<<D_half[i*N+j]<<" ";
		std::cout<<std::endl;	
	} */
	for(int i =0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			Uvector[i*N+j] = S[i*N+j];
	}	
	//U is in column major form.. O = U'DU
    auto D_hfU = matrixmatrixmul(D_half,N,N,Uvector,N,N);
	auto S_half = matrixTmatrixmul(Uvector,N,N,D_hfU,N,N);

	auto phinew = matrixmatrixmul(phi,n_dofs,N,S_half,N,N);

	return phinew;




}