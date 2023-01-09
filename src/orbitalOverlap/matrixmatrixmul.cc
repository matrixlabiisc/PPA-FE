// flattened array naive matrix manipulations

#include "matrixmatrixmul.h"
#include <vector>
#include <iostream>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>

//#ifdef USE_COMPLEX
// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and B is m2 by n2 matrix
std::vector<std::complex<double>>
matrixmatrixmul(const std::vector<std::complex<double>> &A,
                const unsigned int         m1,
                const unsigned int         n1,
                const std::vector<std::complex<double>> &B,
                const unsigned int         m2,
                const unsigned int         n2)
{
  assert((n1 == m2) &&
         "Given matrices are not compatible for matrix multiplication");
  assert((A.size() == m1 * n1) &&
         "First matrix is not compatible with the specified dimensions");
  assert((B.size() == m2 * n2) &&
         "Second matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexB, indexC = 0;

  std::vector<std::complex<double>> C(m1 * n2, std::complex<double> (0.0,0.0)); // initialized to zero

  /*	for (size_t i = 0; i < m1; ++i) // (i+1)th row of A
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
    } */
  char         transA = 'N';
  char         transB = 'N';
  const std::complex<double> alpha(1,0), beta(0,0);
  // std::cout<<"Start Matrix matrix multiplication\n";
  dftfe::zgemm_(&transB,
                &transA,
                &n2,
                &m1,
                &m2,
                &alpha,
                &B[0],
                &n2,
                &A[0],
                &n1,
                &beta,
                &C[0],
                &n2);
    


  return C;
}

// matrix A is m1 by n1 and B is m2 by n2
// this function achieves A^T * B, where both A and B are stored rowwise as a
// vector
std::vector<std::complex<double>>
matrixTmatrixmul(const std::vector<std::complex<double>> &A,
                 const unsigned int         m1,
                 const unsigned int         n1,
                 const std::vector<std::complex<double>> &B,
                 const unsigned int         m2,
                 const unsigned int         n2)
{
  assert((m1 == m2) &&
         "Given matrices are not compatible for matrix multiplication");
  assert((A.size() == m1 * n1) &&
         "First matrix is not compatible with the specified dimensions");
  assert((B.size() == m2 * n2) &&
         "Second matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexB, indexC = 0;

  std::vector<std::complex<double>> C(n1 * n2, std::complex<double> (0.0,0.0)); // initialized to zero
  std::vector<std::complex<double>> Aconj(m1 * n1, std::complex<double> (0.0,0.0));
  for (int i = 0; i < m1; i++)
  {
    for (int j = 0; j < n1; j++ )
      Aconj[i*n1+j] = std::conj(A[i*n1+j]);
  }

  char         transA = 'T';
  char         transB = 'N';
  const std::complex<double> alpha = 1.0, beta = 0.0;
  // std::cout<<"Start MatrixT matrix multiplication\n";
  dftfe::zgemm_(&transB,
                &transA,
                &n2,
                &n1,
                &m1,
                &alpha,
                &B[0],
                &n2,
                &Aconj[0],
                &n1,
                &beta,
                &C[0],
                &n2);

  // std::cout << '\n';
  return C;
}
// matrix A is m1 by n1 and B is m2 by n2
// this function achieves A^T * B, where both A and B are stored rowwise as a
// vector this function is for cases where same matrix is transpose and
// multiplied in this case the result is a symmetric matrix so we store only
// upper triangular part
std::vector<std::complex<double>>
selfMatrixTmatrixmul(const std::vector<std::complex<double>> &A,
                     const unsigned int         m,
                     const unsigned int         n)
{
  assert((A.size() == m * n) &&
         "Given matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexB, indexC = 0;

  std::vector<std::complex<double>> C_upper((n * (n + 1)) / 2, std::complex<double> (0.0,0.0)); // initialized to zero
  std::vector<std::complex<double>> C(n * n, std::complex<double> (0.0,0.0));


  std::vector<std::complex<double>> Aconj(m * n, std::complex<double> (0.0,0.0));
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++ )
      {
        Aconj[i*n+j] = std::conj(A[i*n+j]);

      }

  }


  char         transA = 'T';
  char         transB = 'N';
  const std::complex<double> alpha = 1.0, beta = 0.0;

  dftfe::zgemm_(&transB,
                &transA,
                &n,
                &n,
                &m,
                &alpha,
                &A[0],
                &n,
                &Aconj[0],
                &n,
                &beta,
                &C[0],
                &n);






  int count = 0;
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
        {
          if (i <= j)
            {
              C_upper[count] = C[i * n + j];
              count++;
            }

        }

    }

  return C_upper;
}
// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and BT is n2 by m2 matrix i.e. B is m2 by n2 matrix
// we pass B to the function which has been stored row wise
// or equivalently B is stored as a vector column wise
// and A*B is evaluated efficiently
std::vector<std::complex<double>>
matrixmatrixTmul(const std::vector<std::complex<double>> &A,
                 unsigned int               m1,
                 unsigned int               n1,
                 const std::vector<std::complex<double>> &B,
                 unsigned int               m2,
                 unsigned int               n2)
{
  assert((n1 == n2) &&
         "Given matrices are not compatible for matrix multiplication");
  assert((A.size() == m1 * n1) &&
         "First matrix is not compatible with the specified dimensions");
  assert((B.size() == m2 * n2) &&
         "Second matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexBT, indexC = 0;

  std::vector<std::complex<double>> C(m1 * m2, std::complex<double> (0.0,0.0)); // initialized to zero

  /*for (size_t i = 0; i < m1; ++i) // (i+1)th row of A
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
  } */
  std::vector<std::complex<double>> Bconj(m2 * n2, std::complex<double> (0.0,0.0));
  for (int i = 0; i < m2; i++)
  {
    for (int j = 0; j < n2; j++ )
      Bconj[i*n2+j] = std::conj(B[i*n2+j]);
  }

  char         transA = 'N';
  char         transB = 'T';
  const std::complex<double> alpha = 1.0, beta = 0.0;

  dftfe::zgemm_(&transB,
                &transA,
                &m2,
                &m1,
                &n2,
                &alpha,
                &Bconj[0],
                &n2,
                &A[0],
                &n1,
                &beta,
                &C[0],
                &m2);





  return C;
  
}
std::vector<std::complex<double>>
OrthonormalizationofProjectedWavefn(const std::vector<std::complex<double>> &Sold,
                                    const unsigned int         m1,
                                    const unsigned int         n1,
                                    const std::vector<std::complex<double>> &C,
                                    const unsigned int         m2,
                                    const unsigned int         n2)
{
  // m1: Basis Dimension
  // n1: Total number of Atomic Orbitals
  // m2: Total Number of Atomic Orbitals
  // n2: Number of KS orbitals.
  assert((m2 == n1) && "Number of Atomic Orbitals not consistent");


  // std::cout<<"#Begin OrthoNormalization"<<std::endl;
  std::vector<std::complex<double>> S(m1 * n1, std::complex<double> (0.0,0.0));
  int                 count = 0;
  for (int i = 0; i < m1; i++)
    {
      for (int j = i; j < n1; j++)
        {
          S[i * n1 + j] = Sold[count];
          S[j * m1 + i] = Sold[count];
          count++;
        }
    }

  // std::cout<<"#B=CtS"<<std::endl;
  auto B = matrixTmatrixmul(C, m2, n2, S, m1, n1);
  /*	for (int i = 0; i < n2; i++)
  {
    for(int j = 0; j < n1; j++)
      std::cout<<B[i*n1+j]<<" ";
    std::cout<<std::endl;
  }
  std::cout<<"#O=(CtS)C"<<std::endl; */
  auto O = matrixmatrixmul(B, n2, n1, C, m2, n2);
  /*	for (int i = 0; i < n2; i++)
  {
    for(int j = 0; j < n2; j++)
      std::cout<<O[i*n2+j]<<" ";
    std::cout<<std::endl;
  } */
  const unsigned int N = n2;
  // double D[N];
  std::vector<double> D(N, 0.0);
  count = 0;
  // double upperO[N*N];
  std::vector<std::complex<double>> upperO(N * N, 0.0);
  // std::cout<<"#Begin Upper Triangle creation"<<std::endl;
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          if (j >= i)
            upperO[i * N + j] = O[i * N + j];
          else
            upperO[i * N + j] = 0.0;

        }
    } /*
     for (int i = 0; i < n2; i++)
     {
       for(int j = 0; j < n2; j++)
         std::cout<<upperO[i*n2+j]<<" ";
       std::cout<<std::endl;
     }
     std::cout<<"#Completed Upper Triangle creation"<<std::endl;

         std::cout<<"#Begin Eigen Value Decomposition"<<std::endl; */
      int                info;
      const unsigned int lwork = 1 + 6 * N +
                                 2 * N * N,
                         liwork = 3 + 5 * N;
      std::vector<int>   iwork(liwork, 0);
      const char         jobz = 'V', uplo = 'L';
      const unsigned int lrwork =
        1 + 5 * N + 2 * N * N;
      std::vector<double>               rwork(lrwork);
      std::vector<std::complex<double>> work(lwork);

      dftfe::zheevd_(&jobz,
              &uplo,
              &N,
              &upperO[0],
              &N,
              &D[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);


  //
  // free up memory associated with work
  //
  work.clear();
  iwork.clear();
  std::vector<std::complex<double>>().swap(work);
  std::vector<int>().swap(iwork);
  if (info > 0)
    std::cout << "Eigen Value Decomposition Falied!!" << std::endl;

  std::vector<std::complex<double>> D_half(N * N, std::complex<double> (0.0,0.0));
  std::vector<std::complex<double>> upperOvector(N * N, std::complex<double> (0.0,0.0));
  for (int i = 0; i < N; i++)
    D_half[i * N + i].real(pow(D[i], -0.5)) ;

  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        upperOvector[i * N + j] = upperO[i * N + j];
    }
  // U is in column major form.. O = U'DU
  auto D_hfU  = matrixmatrixmul(D_half, N, N, upperOvector, N, N);
  auto O_half = matrixTmatrixmul(upperOvector, N, N, D_hfU, N, N);

  auto Cnew = matrixmatrixmul(C, m2, n2, O_half, n2, n2);

  auto temp  = matrixTmatrixmul(Cnew, n2, n2, S, m1, n1);
  auto temp2 = matrixmatrixmul(temp, m2, n1, Cnew, m2, n2);



  return Cnew;
}
std::vector<std::complex<double>>
LowdenOrtho(const std::vector<std::complex<double>> &phi,
            int                        n_dofs,
            int                        N,
            const std::vector<std::complex<double>> &UpperS)
{
  std::vector<std::complex<double>> S(N * N, std::complex<double> (0.0,0.0));
  int                 count = 0;
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          S[i * N + j] = 0.0;
          if (j >= i)
            {
              S[i * N + j] = UpperS[count];
              count++;
            }
        }
    }



  const unsigned int  Nrow = N;
      int                info;
      const unsigned int lwork = 1 + 6 * N +
                                 2 * N * N,
                         liwork = 3 + 5 * N;
      std::vector<int>   iwork(liwork, 0);
      const char         jobz = 'V', uplo = 'L';
      const unsigned int lrwork =
        1 + 5 * N + 2 * N * N;
      std::vector<double>               rwork(lrwork);
      std::vector<std::complex<double>> work(lwork);
      std::vector<double> D(N, 0.0);

      dftfe::zheevd_(&jobz,
              &uplo,
              &Nrow,
              &S[0],
              &Nrow,
              &D[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);


  //
  // free up memory associated with work
  //
  work.clear();
  iwork.clear();
  std::vector<std::complex<double>>().swap(work);
  std::vector<int>().swap(iwork);
  if (info > 0)
    std::cout << "Eigen Value Decomposition Falied!!" << std::endl;

  std::vector<std::complex<double>> D_half(N * N, std::complex<double> (0.0,0.0));
  for (int i = 0; i < N; i++)
  {
    D_half[i * N + i].real(pow(D[i], -0.5))  ;

  } 

  // U is in column major form.. O = U'DU
  auto D_hfU  = matrixmatrixmul(D_half, N, N, S, N, N);
  auto S_half = matrixTmatrixmul(S, N, N, D_hfU, N, N);

  auto phinew = matrixmatrixmul(phi, n_dofs, N, S_half, N, N);

  return phinew;
}
std::vector<std::complex<double>>
powerOfMatrix(double power, const std::vector<double> &D, const std::vector<std::complex<double>> &U,const unsigned int N, std::vector<std::complex<double>> Umod)
{

  std::vector<std::complex<double>> Dpow(N*N,std::complex<double> (0.0,0.0));
  for(int i = 0; i < N; i++)
    Dpow[i*N+i].real(pow(D[i],power)) ;
  //dftfe::dlascl2_(&N,&N,&Dpow[0],&Umod[0],&N);
  auto temp = matrixmatrixTmul(Dpow,N,N,U,N,N);
  auto Newmatrix = matrixmatrixmul(U,N,N,temp,N,N);


  return Newmatrix;
}

std::vector<std::complex<double>> diagonalization(std::vector<std::complex<double>> &S, int N, std::vector<double> &D)
{
  

 
  std::vector<std::complex<double>> Stemp = S;
  
  const unsigned int  Nrow = N;
      int                info;
      const unsigned int lwork = 1 + 6 * N +
                                 2 * N * N,
                         liwork = 3 + 5 * N;
      std::vector<int>   iwork(liwork, 0);
      const char         jobz = 'V', uplo = 'L';
      const unsigned int lrwork =
        1 + 5 * N + 2 * N * N;
      std::vector<double>               rwork(lrwork);
      std::vector<std::complex<double>> work(lwork);


      dftfe::zheevd_(&jobz,
              &uplo,
              &Nrow,
              &Stemp[0],
              &Nrow,
              &D[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);


  //
  // free up memory associated with work
  //
      work.clear();
      iwork.clear();
      rwork.clear();
      std::vector<std::complex<double>>().swap(work);
      std::vector<double>().swap(rwork);
      std::vector<int>().swap(iwork);
  std::cout<<"Info Value: "<<info<<std::endl;    
  if (info > 0)
    std::cout << "Eigen Value Decomposition Falied!!" << std::endl;  

  return Stemp;

}

std::vector<std::complex<double>> 
TransposeMatrix(std::vector<std::complex<double>> &A, int N)
{
  std::vector<std::complex<double>> B(N*N,std::complex<double> (0.0,0.0));
  
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
      B[i*N+j] = std::conj(A[j*N+i]);
  }
  
  return B;

}
std::vector<std::complex<double>>
computeHprojOrbital(std::vector<std::complex<double>>              C,
                    std::vector<std::complex<double>> &C_hat,
                    const unsigned int                              m,
                    const unsigned int                             N,
                    std::vector<double> &H)
{

  std::vector<std::complex<double>> Hmatrix(N*N,std::complex<double> (0.0,0.0));
  for(int i = 0; i < N; i++)
    Hmatrix[i*N+i].real(H[i])  ;
  auto temp = matrixmatrixTmul(Hmatrix,N,N,C_hat,m,N);
  auto Hproj  = matrixmatrixmul(C_hat, m, N, temp, N, m);


  return Hproj;
}


//#else
std::vector<double>
inverseOfOverlapMatrix(const std::vector<double> &SmatrixVec,
                       const size_t               matrixDim)
{
  /*	std::vector<double> invS(matrixDim * matrixDim, 0.0); // storing full
    inverse matrix

    Eigen::MatrixXd S(matrixDim, matrixDim);

    unsigned int count = 0;

    for (unsigned int i = 0; i < matrixDim; ++i)
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
      for (unsigned int j = 0; j < matrixDim; ++j)
      {
        // invS[count] = invS_eigen(i, j); // rowwise storage
        invS[count] = invS_direct(i, j);
        ++count;
      }
    }
    */
  std::vector<double> S(matrixDim * matrixDim, 0.0);
  int                 count = 0;
  for (int i = 0; i < matrixDim; i++)
    {
      for (int j = 0; j < matrixDim; j++)
        {
          if (j >= i)
            {
              S[i * matrixDim + j] = SmatrixVec[count];
              count++;
            }
        }
    }


  int                 info;
  const unsigned int  N     = matrixDim;
  const unsigned int  lwork = 1 + 6 * N + 2 * N * N, liwork = 3 + 5 * N;
  std::vector<int>    iwork(liwork, 0);
  const char          jobz = 'V', uplo = 'L';
  std::vector<double> work(lwork);
  std::vector<double> D(N, 0.0);
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
  if (info > 0)
    std::cout << "Eigen Value Decomposition Failed!!" << std::endl;
  /* if(dftfe::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
  }	*/



  std::vector<double> D_inv(N * N, 0.0);
  std::vector<double> Uvector(N * N, 0);
  for (int i = 0; i < N; i++)
    D_inv[i * N + i] = pow(D[i], -1.0);

  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        Uvector[i * N + j] = S[i * N + j];
    }
  // U is in column major form.. O = U'DU
  auto D_invU = matrixmatrixmul(D_inv, N, N, Uvector, N, N);
  auto invS   = matrixTmatrixmul(Uvector, N, N, D_invU, N, N);
  return invS;
}

// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and B is m2 by n2 matrix
std::vector<double>
matrixmatrixmul(const std::vector<double> &A,
                const unsigned int         m1,
                const unsigned int         n1,
                const std::vector<double> &B,
                const unsigned int         m2,
                const unsigned int         n2)
{
  assert((n1 == m2) &&
         "Given matrices are not compatible for matrix multiplication");
  assert((A.size() == m1 * n1) &&
         "First matrix is not compatible with the specified dimensions");
  assert((B.size() == m2 * n2) &&
         "Second matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexB, indexC = 0;

  std::vector<double> C(m1 * n2, 0.0); // initialized to zero

  /*	for (size_t i = 0; i < m1; ++i) // (i+1)th row of A
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
    } */
  char         transA = 'N';
  char         transB = 'N';
  const double alpha = 1.0, beta = 0.0;
  // std::cout<<"Start Matrix matrix multiplication\n";
  dftfe::dgemm_(&transB,
                &transA,
                &n2,
                &m1,
                &m2,
                &alpha,
                &B[0],
                &n2,
                &A[0],
                &n1,
                &beta,
                &C[0],
                &n2);



  return C;
}
// matrix A is m1 by n1 and B is m2 by n2
// this function achieves A^T * B, where both A and B are stored rowwise as a
// vector
std::vector<double>
matrixTmatrixmul(const std::vector<double> &A,
                 const unsigned int         m1,
                 const unsigned int         n1,
                 const std::vector<double> &B,
                 const unsigned int         m2,
                 const unsigned int         n2)
{
  assert((m1 == m2) &&
         "Given matrices are not compatible for matrix multiplication");
  assert((A.size() == m1 * n1) &&
         "First matrix is not compatible with the specified dimensions");
  assert((B.size() == m2 * n2) &&
         "Second matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexB, indexC = 0;

  std::vector<double> C(n1 * n2, 0.0); // initialized to zero



  /*	for (size_t i = 0; i < n1; ++i) // (i+1)th column of A or row of A^T
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
    } */

  char         transA = 'T';
  char         transB = 'N';
  const double alpha = 1.0, beta = 0.0;
  // std::cout<<"Start MatrixT matrix multiplication\n";
  dftfe::dgemm_(&transB,
                &transA,
                &n2,
                &n1,
                &m1,
                &alpha,
                &B[0],
                &n2,
                &A[0],
                &n1,
                &beta,
                &C[0],
                &n2);


  return C;
}
// matrix A is m1 by n1 and B is m2 by n2
// this function achieves A^T * B, where both A and B are stored rowwise as a
// vector this function is for cases where same matrix is transpose and
// multiplied in this case the result is a symmetric matrix so we store only
// upper triangular part
std::vector<double>
selfMatrixTmatrixmul(const std::vector<double> &A,
                     const unsigned int         m,
                     const unsigned int         n)
{
  assert((A.size() == m * n) &&
         "Given matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexB, indexC = 0;

  std::vector<double> C_upper((n * (n + 1)) / 2, 0.0); // initialized to zero
  std::vector<double> C(n * n, 0.0);



  char         transA = 'T';
  char         transB = 'N';
  const double alpha = 1.0, beta = 0.0;

  dftfe::dgemm_(&transB,
                &transA,
                &n,
                &n,
                &m,
                &alpha,
                &A[0],
                &n,
                &A[0],
                &n,
                &beta,
                &C[0],
                &n);





  int count = 0;
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
        {
          if (i <= j)
            {
              C_upper[count] = C[i * n + j];
              count++;
            }

        }

    }

  return C_upper;
}
// both matrices are full matrices written as rowwise flattened vectors
// A is m1 by n1 matrix, and BT is n2 by m2 matrix i.e. B is m2 by n2 matrix
// we pass B to the function which has been stored row wise
// or equivalently B is stored as a vector column wise
// and A*B is evaluated efficiently
std::vector<double>
matrixmatrixTmul(const std::vector<double> &A,
                 unsigned int               m1,
                 unsigned int               n1,
                 const std::vector<double> &B,
                 unsigned int               m2,
                 unsigned int               n2)
{
  assert((n1 == n2) &&
         "Given matrices are not compatible for matrix multiplication");
  assert((A.size() == m1 * n1) &&
         "First matrix is not compatible with the specified dimensions");
  assert((B.size() == m2 * n2) &&
         "Second matrix is not compatible with the specified dimensions");

  // unsigned int indexA, indexBT, indexC = 0;

  std::vector<double> C(m1 * m2, 0.0); // initialized to zero

  /*for (size_t i = 0; i < m1; ++i) // (i+1)th row of A
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
  } */
  char         transA = 'N';
  char         transB = 'T';
  const double alpha = 1.0, beta = 0.0;

  dftfe::dgemm_(&transB,
                &transA,
                &m2,
                &m1,
                &n2,
                &alpha,
                &B[0],
                &n2,
                &A[0],
                &n1,
                &beta,
                &C[0],
                &m2);




  return C;
}
std::vector<double>
OrthonormalizationofProjectedWavefn(const std::vector<double> &Sold,
                                    const unsigned int         m1,
                                    const unsigned int         n1,
                                    const std::vector<double> &C,
                                    const unsigned int         m2,
                                    const unsigned int         n2)
{
  // m1: Basis Dimension
  // n1: Total number of Atomic Orbitals
  // m2: Total Number of Atomic Orbitals
  // n2: Number of KS orbitals.
  assert((m2 == n1) && "Number of Atomic Orbitals not consistent");


  // std::cout<<"#Begin OrthoNormalization"<<std::endl;
  std::vector<double> S(m1 * n1, 0.0);
  int                 count = 0;
  for (int i = 0; i < m1; i++)
    {
      for (int j = i; j < n1; j++)
        {
          S[i * n1 + j] = Sold[count];
          S[j * m1 + i] = Sold[count];
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
  // std::cout<<"#B=CtS"<<std::endl;
  auto B = matrixTmatrixmul(C, m2, n2, S, m1, n1);
  /*	for (int i = 0; i < n2; i++)
  {
    for(int j = 0; j < n1; j++)
      std::cout<<B[i*n1+j]<<" ";
    std::cout<<std::endl;
  }
  std::cout<<"#O=(CtS)C"<<std::endl; */
  auto O = matrixmatrixmul(B, n2, n1, C, m2, n2);
  /*	for (int i = 0; i < n2; i++)
  {
    for(int j = 0; j < n2; j++)
      std::cout<<O[i*n2+j]<<" ";
    std::cout<<std::endl;
  } */
  const unsigned int N = n2;
  // double D[N];
  std::vector<double> D(N, 0.0);
  count = 0;
  // double upperO[N*N];
  std::vector<double> upperO(N * N, 0.0);
  // std::cout<<"#Begin Upper Triangle creation"<<std::endl;
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          if (j >= i)
            upperO[i * N + j] = O[i * N + j];
          else
            upperO[i * N + j] = 0.0;
        }
    } /*
     for (int i = 0; i < n2; i++)
     {
       for(int j = 0; j < n2; j++)
         std::cout<<upperO[i*n2+j]<<" ";
       std::cout<<std::endl;
     }
     std::cout<<"#Completed Upper Triangle creation"<<std::endl;

         std::cout<<"#Begin Eigen Value Decomposition"<<std::endl; */
  int                 info;
  const unsigned int  lwork = 1 + 6 * N + 2 * N * N, liwork = 3 + 5 * N;
  std::vector<int>    iwork(liwork, 0);
  const char          jobz = 'V', uplo = 'L';
  std::vector<double> work(lwork);

  dftfe::dsyevd_(&jobz,
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
  if (info > 0)
    std::cout << "Eigen Value Decomposition Falied!!" << std::endl;
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
  std::vector<double> D_half(N * N, 0.0);
  std::vector<double> upperOvector(N * N, 0);
  for (int i = 0; i < N; i++)
    D_half[i * N + i] = pow(D[i], -0.5);
  /*std::cout<<"D half matrix: "<<std::endl;
  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < N; j++)
      std::cout<<D_half[i*N+j]<<" ";
    std::cout<<std::endl;
  } */
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        upperOvector[i * N + j] = upperO[i * N + j];
    }
  // U is in column major form.. O = U'DU
  auto D_hfU  = matrixmatrixmul(D_half, N, N, upperOvector, N, N);
  auto O_half = matrixTmatrixmul(upperOvector, N, N, D_hfU, N, N);

  auto Cnew = matrixmatrixmul(C, m2, n2, O_half, n2, n2);

  auto temp  = matrixTmatrixmul(Cnew, n2, n2, S, m1, n1);
  auto temp2 = matrixmatrixmul(temp, m2, n1, Cnew, m2, n2);
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
std::vector<double>
InvertPowerMatrix(double power, int N, const std::vector<double> &UpperS)
{
  std::vector<double> S(N * N, 0.0);
  int                 count = 0;
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          S[i * N + j] = 0.0;
          if (j >= i)
            {
              S[i * N + j] = UpperS[count];
              count++;
            }
        }
    }
  // std::cout<<"#Begin Eigen Value Decomposition"<<std::endl;
  const unsigned int  Nrow = N;
  int                 info;
  const unsigned int  lwork = 1 + 6 * N + 2 * N * N, liwork = 3 + 5 * N;
  std::vector<int>    iwork(liwork, 0);
  const char          jobz = 'V', uplo = 'L';
  std::vector<double> work(lwork);
  std::vector<double> D(N, 0.0);
  dftfe::dsyevd_(&jobz,
                 &uplo,
                 &Nrow,
                 &S[0],
                 &Nrow,
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
  if (info > 0)
    std::cout << "Eigen Value Decomposition Falied!!" << std::endl;

  std::vector<double> D_half(N * N, 0.0);
  for (int i = 0; i < N; i++)
    D_half[i * N + i] = pow(D[i], power);

  // U is in column major form.. O = U'DU
  auto D_hfU  = matrixmatrixmul(D_half, N, N, S, N, N);
  auto S_half = matrixTmatrixmul(S, N, N, D_hfU, N, N);
  return S_half;
}



std::vector<double>
LowdenOrtho(const std::vector<double> &phi,
            int                        n_dofs,
            int                        N,
            const std::vector<double> &UpperS)
{
  std::vector<double> S(N * N, 0.0);
  int                 count = 0;
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          S[i * N + j] = 0.0;
          if (j >= i)
            {
              S[i * N + j] = UpperS[count];
              count++;
            }
        }
    }
  // std::cout<<"#Begin Eigen Value Decomposition"<<std::endl;
  const unsigned int  Nrow = N;
  int                 info;
  const unsigned int  lwork = 1 + 6 * N + 2 * N * N, liwork = 3 + 5 * N;
  std::vector<int>    iwork(liwork, 0);
  const char          jobz = 'V', uplo = 'L';
  std::vector<double> work(lwork);
  std::vector<double> D(N, 0.0);
  dftfe::dsyevd_(&jobz,
                 &uplo,
                 &Nrow,
                 &S[0],
                 &Nrow,
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
  if (info > 0)
    std::cout << "Eigen Value Decomposition Falied!!" << std::endl;

  std::vector<double> D_half(N * N, 0.0);
  for (int i = 0; i < N; i++)
    D_half[i * N + i] = pow(D[i], -0.5);

  // U is in column major form.. O = U'DU
  auto D_hfU  = matrixmatrixmul(D_half, N, N, S, N, N);
  auto S_half = matrixTmatrixmul(S, N, N, D_hfU, N, N);

  auto phinew = matrixmatrixmul(phi, n_dofs, N, S_half, N, N);

  return phinew;
}
std::vector<double>
computeHprojOrbital(std::vector<double>              C,
                    std::vector<double> &C_hat,
                    const unsigned int                              m,
                    const unsigned int                             N,
                    std::vector<double> &H)
{


  dftfe::dlascl2_(&N,&m,&H[0],&C[0],&N);

  auto Hproj  = matrixmatrixTmul(C_hat, m, N, C, N, m);


  return Hproj;
}




std::vector<double>
powerOfMatrix(double power, const std::vector<double> &D, const std::vector<double> &U,const unsigned int N, std::vector<double> Umod)
{

  std::vector<double> Dpow(N,0.0);
  for(int i = 0; i < N; i++)
    Dpow[i] = pow(D[i],power);
  dftfe::dlascl2_(&N,&N,&Dpow[0],&Umod[0],&N);
  //auto temp = matrixmatrixmul(Dpow,N,N,U,N,N);
  auto Newmatrix = matrixmatrixTmul(U,N,N,Umod,N,N);


  return Newmatrix;
}
std::vector<double> diagonalization(std::vector<double> S, int N, std::vector<double> &D)
{
  

 
  
  
  const unsigned int  Nrow = N;
  int                 info;
  const unsigned int  lwork = 1 + 6 * N + 2 * N * N, liwork = 3 + 5 * N;
  std::vector<int>    iwork(liwork, 0);
  const char          jobz = 'V', uplo = 'L';
  std::vector<double> work(lwork);
  dftfe::dsyevd_(&jobz,
                 &uplo,
                 &Nrow,
                 &S[0],
                 &Nrow,
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
  if (info > 0)
    std::cout << "Eigen Value Decomposition Falied!!" << std::endl;  

  return S;

}
std::vector<double> 
TransposeMatrix(std::vector<double> &A, int N)
{
  std::vector<double> B(N*N,0.0);
  
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
      B[i*N+j] = A[j*N+i];
  }
  
  return B;

}
//#endif





























