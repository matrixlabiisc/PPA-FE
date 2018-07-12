// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Phani Motamarri, Sambit Das
//


/** @file linearAlgebraOperationsOpt.cc
 *  @brief Contains linear algebra operations
 *
 */

#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <dftParameters.h>
#include <dftUtils.h>

#include "pseudoGS.cc"

namespace dftfe{

  namespace linearAlgebraOperations
  {

    void callevd(const unsigned int dimensionMatrix,
		 double *matrix,
		 double *eigenValues)
    {

      int info;
      const unsigned int lwork = 1 + 6*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix, liwork = 3 + 5*dimensionMatrix;
      std::vector<int> iwork(liwork,0);
      const char jobz='V', uplo='U';
      std::vector<double> work(lwork);

      dsyevd_(&jobz,
	      &uplo,
	      &dimensionMatrix,
	      matrix,
	      &dimensionMatrix,
	      eigenValues,
	      &work[0],
	      &lwork,
	      &iwork[0],
	      &liwork,
	      &info);

      //
      //free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<double>().swap(work);
      std::vector<int>().swap(iwork);

    }


     void callevd(const unsigned int dimensionMatrix,
		 std::complex<double> *matrix,
		 double *eigenValues)
    {
      int info;
      const unsigned int lwork = 1 + 6*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix, liwork = 3 + 5*dimensionMatrix;
      std::vector<int> iwork(liwork,0);
      const char jobz='V', uplo='U';
      const unsigned int lrwork = 1 + 5*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix;
      std::vector<double> rwork(lrwork);
      std::vector<std::complex<double> > work(lwork);


      zheevd_(&jobz,
	      &uplo,
	      &dimensionMatrix,
	      matrix,
	      &dimensionMatrix,
	      eigenValues,
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);

      //
      //free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<std::complex<double> >().swap(work);
      std::vector<int>().swap(iwork);


    }


    void callevr(const unsigned int dimensionMatrix,
		 std::complex<double> *matrixInput,
		 std::complex<double> *eigenVectorMatrixOutput,
		 double *eigenValues)
    {
      char jobz = 'V', uplo = 'U', range = 'A';
      const double vl=0.0,vu=0.0;
      const unsigned int il=0,iu = 0;
      const double abstol = 1e-08;
      std::vector<unsigned int> isuppz(2*dimensionMatrix);
      const int lwork = 2*dimensionMatrix;
      std::vector<std::complex<double> > work(lwork);
      const int liwork = 10*dimensionMatrix;
      std::vector<int> iwork(liwork);
      const int lrwork = 24*dimensionMatrix;
      std::vector<double> rwork(lrwork);
      int info;

      zheevr_(&jobz,
	      &range,
	      &uplo,
	      &dimensionMatrix,
	      matrixInput,
	      &dimensionMatrix,
	      &vl,
	      &vu,
	      &il,
	      &iu,
	      &abstol,
	      &dimensionMatrix,
	      eigenValues,
	      eigenVectorMatrixOutput,
	      &dimensionMatrix,
	      &isuppz[0],
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);

      AssertThrow(info==0,dealii::ExcMessage("Error in zheevr"));
    }




    void callevr(const unsigned int dimensionMatrix,
		 double *matrixInput,
		 double *eigenVectorMatrixOutput,
		 double *eigenValues)
    {
      char jobz = 'V', uplo = 'U', range = 'A';
      const double vl=0.0,vu = 0.0;
      const unsigned int il=0,iu=0;
      const double abstol = 0.0;
      std::vector<unsigned int> isuppz(2*dimensionMatrix);
      const int lwork = 26*dimensionMatrix;
      std::vector<double> work(lwork);
      const int liwork = 10*dimensionMatrix;
      std::vector<int> iwork(liwork);
      int info;

      dsyevr_(&jobz,
	      &range,
	      &uplo,
	      &dimensionMatrix,
	      matrixInput,
	      &dimensionMatrix,
	      &vl,
	      &vu,
	      &il,
	      &iu,
	      &abstol,
	      &dimensionMatrix,
	      eigenValues,
	      eigenVectorMatrixOutput,
	      &dimensionMatrix,
	      &isuppz[0],
	      &work[0],
	      &lwork,
	      &iwork[0],
	      &liwork,
	      &info);

      AssertThrow(info==0,dealii::ExcMessage("Error in dsyevr"));


    }




    void callgemm(const unsigned int numberEigenValues,
		  const unsigned int localVectorSize,
		  const std::vector<double> & eigenVectorSubspaceMatrix,
		  const dealii::parallel::distributed::Vector<double> & X,
		  dealii::parallel::distributed::Vector<double> & Y)

    {

      const char transA  = 'T', transB  = 'N';
      const double alpha = 1.0, beta = 0.0;
      dgemm_(&transA,
	     &transB,
	     &numberEigenValues,
	     &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &eigenVectorSubspaceMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     Y.begin(),
	     &numberEigenValues);

    }


    void callgemm(const unsigned int numberEigenValues,
		  const unsigned int localVectorSize,
		  const std::vector<std::complex<double> > & eigenVectorSubspaceMatrix,
		  const dealii::parallel::distributed::Vector<std::complex<double> > & X,
		  dealii::parallel::distributed::Vector<std::complex<double> > & Y)

    {

      const char transA  = 'T', transB  = 'N';
      const std::complex<double> alpha = 1.0, beta = 0.0;
      zgemm_(&transA,
	     &transB,
	     &numberEigenValues,
	     &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &eigenVectorSubspaceMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     Y.begin(),
	     &numberEigenValues);

    }



    //
    //chebyshev filtering of given subspace XArray
    //
    template<typename T>
    void chebyshevFilter(operatorDFTClass & operatorMatrix,
			 dealii::parallel::distributed::Vector<T> & XArray,
			 const unsigned int numberWaveFunctions,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0)
    {
      double e, c, sigma, sigma1, sigma2, gamma;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;

      dealii::parallel::distributed::Vector<T> YArray;//,YNewArray;

      //
      //create YArray
      //
      YArray.reinit(XArray);


      //
      //initialize to zeros.
      //x
      const T zeroValue = 0.0;
      YArray = zeroValue;


      //
      //call HX
      //
      bool scaleFlag = false;
      double scalar = 1.0;
      operatorMatrix.HX(XArray,
			numberWaveFunctions,
			scaleFlag,
			scalar,
			YArray);


      double alpha1 = sigma1/e, alpha2 = -c;

      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      YArray.add(alpha2,XArray);
      YArray *= alpha1;

      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);

	  //
	  //multiply XArray with alpha2
	  //
	  XArray *= alpha2;
	  XArray.add(-c*alpha1,YArray);


	  //
	  //call HX
	  //
	  bool scaleFlag = true;
	  operatorMatrix.HX(YArray,
			    numberWaveFunctions,
			    scaleFlag,
			    alpha1,
			    XArray);

	  //
	  //XArray = YArray
	  //
	  XArray.swap(YArray);

	  //
	  //YArray = YNewArray
	  //
	  sigma = sigma2;

	}

      //copy back YArray to XArray
      XArray = YArray;

    }

    template<typename T>
    void gramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
				      const unsigned int numberVectors)
    {

      const unsigned int localVectorSize = X.local_size()/numberVectors;

      //
      //Create template PETSc vector to create BV object later
      //
      Vec templateVec;
      VecCreateMPI(X.get_mpi_communicator(),
		   localVectorSize,
		   PETSC_DETERMINE,
		   &templateVec);
      VecSetFromOptions(templateVec);


      //
      //Set BV options after creating BV object
      //
      BV columnSpaceOfVectors;
      BVCreate(X.get_mpi_communicator(),&columnSpaceOfVectors);
      BVSetSizesFromVec(columnSpaceOfVectors,
			templateVec,
			numberVectors);
      BVSetFromOptions(columnSpaceOfVectors);


      //
      //create list of indices
      //
      std::vector<PetscInt> indices(localVectorSize);
      std::vector<PetscScalar> data(localVectorSize,0.0);

      PetscInt low,high;

      VecGetOwnershipRange(templateVec,
			   &low,
			   &high);


      for(PetscInt index = 0;index < localVectorSize; ++index)
	indices[index] = low+index;

      VecDestroy(&templateVec);

      //
      //Fill in data into BV object
      //
      Vec v;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v);
	  VecSet(v,0.0);
	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    data[iNode] = X.local_element(numberVectors*iNode + iColumn);

	  VecSetValues(v,
		       localVectorSize,
		       &indices[0],
		       &data[0],
		       INSERT_VALUES);

	  VecAssemblyBegin(v);
	  VecAssemblyEnd(v);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v);
	}

      //
      //orthogonalize
      //
      BVOrthogonalize(columnSpaceOfVectors,NULL);

      //
      //Copy data back into X
      //
      Vec v1;
      PetscScalar * pointerv1;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v1);

	  VecGetArray(v1,
		      &pointerv1);

	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    X.local_element(numberVectors*iNode + iColumn) = pointerv1[iNode];

	  VecRestoreArray(v1,
			  &pointerv1);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v1);
	}

      BVDestroy(&columnSpaceOfVectors);

    }

#if(defined DEAL_II_WITH_SCALAPACK && !USE_COMPLEX)
    template<typename T>
    void rayleighRitz(operatorDFTClass & operatorMatrix,
		      dealii::parallel::distributed::Vector<T> & X,
		      const unsigned int numberWaveFunctions,
		      const MPI_Comm &interBandGroupComm,
		      std::vector<double> & eigenValues)

    {
      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ||
					  dftParameters::verbosity<4 ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);
      //
      //compute projected Hamiltonian
      //
      const unsigned rowsBlockSize=std::min((unsigned int)50,numberWaveFunctions);
      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
      internal::createProcessGridSquareMatrix(X.get_mpi_communicator(),
		                              numberWaveFunctions,
					      processGrid);

      dealii::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                            processGrid,
                                            rowsBlockSize);

      computing_timer.enter_section("Blocked XtHX, RR step");
      operatorMatrix.XtHX(X,
			  numberWaveFunctions,
			  processGrid,
			  projHamPar);
      computing_timer.exit_section("Blocked XtHX, RR step");

      //
      //compute eigendecomposition of ProjHam
      //
      computing_timer.enter_section("ScaLAPACK eigen decomp, RR step");
      const unsigned int numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);
      eigenValues=projHamPar.eigenpairs_symmetric_by_index_MRRR(std::make_pair(0,numberWaveFunctions-1),true);
      computing_timer.exit_section("ScaLAPACK eigen decomp, RR step");

      computing_timer.enter_section("Broadcast eigvec across band groups, RR step");
      internal::broadcastAcrossInterCommScaLAPACKMat
	                                   (processGrid,
		                            projHamPar,
				            interBandGroupComm,
					    0);
      computing_timer.exit_section("Broadcast eigvec across band groups, RR step");
      //
      //rotate the basis in the subspace X = X*Q, implemented as X^{T}=Q^{T}*X^{T} with X^{T}
      //stored in the column major format
      //
      computing_timer.enter_section("Blocked subspace rotation, RR step");
      internal::subspaceRotation(X,
		                 numberWaveFunctions,
		                 processGrid,
				 interBandGroupComm,
			         projHamPar,
				 true);
      computing_timer.exit_section("Blocked subspace rotation, RR step");

    }
#else

    template<typename T>
    void rayleighRitz(operatorDFTClass & operatorMatrix,
		      dealii::parallel::distributed::Vector<T> & X,
		      const unsigned int numberWaveFunctions,
		      const MPI_Comm &interBandGroupComm,
		      std::vector<double> & eigenValues)
    {
      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ||
					  dftParameters::verbosity<4 ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);
      //
      //compute projected Hamiltonian
      //
      std::vector<T> ProjHam;
      const unsigned int numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);

      computing_timer.enter_section("XtHX");
      operatorMatrix.XtHX(X,
			  numberEigenValues,
			  ProjHam);
      computing_timer.exit_section("XtHX");

      //
      //compute eigendecomposition of ProjHam
      //
      computing_timer.enter_section("eigen decomp in RR");
      callevd(numberEigenValues,
	      &ProjHam[0],
	      &eigenValues[0]);

#ifdef USE_COMPLEX
      MPI_Bcast(&ProjHam[0],
	        numberEigenValues*numberEigenValues,
                MPI_C_DOUBLE_COMPLEX,
	        0,
	        X.get_mpi_communicator());
#else
      MPI_Bcast(&ProjHam[0],
	        numberEigenValues*numberEigenValues,
	        MPI_DOUBLE,
	        0,
	        X.get_mpi_communicator());
#endif

      computing_timer.exit_section("eigen decomp in RR");


      //
      //rotate the basis in the subspace X = X*Q
      //
      const unsigned int localVectorSize = X.local_size()/numberEigenValues;
      dealii::parallel::distributed::Vector<T> rotatedBasis;
      rotatedBasis.reinit(X);

      computing_timer.enter_section("subspace rotation in RR");
      callgemm(numberEigenValues,
	       localVectorSize,
	       ProjHam,
	       X,
	       rotatedBasis);
      computing_timer.exit_section("subspace rotation in RR");

      X = rotatedBasis;
    }
#endif

#ifdef DEAL_II_WITH_SCALAPACK
    template<typename T>
    void computeEigenResidualNorm(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<T> & X,
				  const std::vector<double> & eigenValues,
				  std::vector<double> & residualNorm)

    {

      //
      //get the number of eigenVectors
      //
      const unsigned int totalNumberVectors = eigenValues.size();
      const unsigned int localVectorSize = X.local_size()/totalNumberVectors;
      std::vector<double> residualNormSquare(totalNumberVectors,0.0);

      //create temporary arrays XBlock,HXBlock
      dealii::parallel::distributed::Vector<dataTypes::number> XBlock,HXBlock;

      // Do H*X using a blocked approach and compute
      // the residual norms: H*XBlock-XBlock*D, where
      // D is the eigenvalues matrix.
      // The blocked approach avoids additional full
      // wavefunction matrix memory
      const unsigned int vectorsBlockSize=dftParameters::orthoRRWaveFuncBlockSize;
      for (unsigned int jvec = 0; jvec < totalNumberVectors; jvec += vectorsBlockSize)
      {
	  // Correct block dimensions if block "goes off edge"
	  const unsigned int B = std::min(vectorsBlockSize, totalNumberVectors-jvec);
	  if (jvec==0 || B!=vectorsBlockSize)
	  {
	     operatorMatrix.reinit(B,
		                   XBlock,
		                   true);
	     HXBlock.reinit(XBlock);
	  }

          XBlock=T(0.);
	  //fill XBlock from X:
	  for(unsigned int iNode = 0; iNode<localVectorSize; ++iNode)
	      for(unsigned int iWave = 0; iWave < B; ++iWave)
		    XBlock.local_element(iNode*B
			     +iWave)
			 =X.local_element(iNode*totalNumberVectors+jvec+iWave);


	  //evaluate H times XBlock and store in HXBlock
	  HXBlock=T(0.);
	  const bool scaleFlag = false;
	  const double scalar = 1.0;
	  operatorMatrix.HX(XBlock,
	                    B,
	                    scaleFlag,
	                    scalar,
	                    HXBlock);

	  //compute residual norms:
	  for(unsigned int iDof = 0; iDof < localVectorSize; ++iDof)
	      for(unsigned int iWave = 0; iWave < B; iWave++)
		{
		  const double temp =std::abs(HXBlock.local_element(B*iDof + iWave) -
		      eigenValues[jvec+iWave]*XBlock.local_element(B*iDof + iWave));
		  residualNormSquare[jvec+iWave] += temp*temp;
		}
      }


      dealii::Utilities::MPI::sum(residualNormSquare,X.get_mpi_communicator(),residualNormSquare);
      if(dftParameters::verbosity>=4)
	{
	  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	    std::cout<<"L-2 Norm of residue   :"<<std::endl;
	}
      for(unsigned int iWave = 0; iWave < totalNumberVectors; ++iWave)
	{
	  residualNorm[iWave] = sqrt(residualNormSquare[iWave]);

	  if(dftParameters::verbosity>=4)
	      if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		std::cout<<"eigen vector "<< iWave<<": "<<residualNorm[iWave]<<std::endl;
	}

      if(dftParameters::verbosity>=4)
	if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	  std::cout <<std::endl;

    }
#else
    template<typename T>
    void computeEigenResidualNorm(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<T> & X,
				  const std::vector<double> & eigenValues,
				  std::vector<double> & residualNorm)

    {

      //
      //get the number of eigenVectors
      //
      const unsigned int totalNumberVectors = eigenValues.size();
      std::vector<double> residualNormSquare(totalNumberVectors,0.0);

      //
      //reinit blockSize require for HX later
      //
      operatorMatrix.reinit(totalNumberVectors,
	      		    X,
		            false);

      //
      //create temp Array
      //
      dealii::parallel::distributed::Vector<T> Y;
      Y.reinit(X);

      //
      //initialize to zero
      //
      const T zeroValue = 0.0;
      Y = zeroValue;

      //
      //compute operator times X
      //
      bool scaleFlag = false;
      T scalar = 1.0;
      operatorMatrix.HX(X,
			totalNumberVectors,
			scaleFlag,
			scalar,
			Y);

      if(dftParameters::verbosity>=4)
	{
	  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	    std::cout<<"L-2 Norm of residue   :"<<std::endl;
	}


      const unsigned int localVectorSize = X.local_size()/totalNumberVectors;

      //
      //compute residual norms
      //
      for(unsigned int iDof = 0; iDof < localVectorSize; ++iDof)
	{
	  for(unsigned int iWave = 0; iWave < totalNumberVectors; iWave++)
	    {
	      T value = Y.local_element(totalNumberVectors*iDof + iWave) - eigenValues[iWave]*X.local_element(totalNumberVectors*iDof + iWave);
	      residualNormSquare[iWave] += std::abs(value)*std::abs(value);
	    }
	}


      dealii::Utilities::MPI::sum(residualNormSquare,X.get_mpi_communicator(),residualNormSquare);
      for(unsigned int iWave = 0; iWave < totalNumberVectors; ++iWave)
	{
	  residualNorm[iWave] = sqrt(residualNormSquare[iWave]);

	  if(dftParameters::verbosity>=4)
	    {
	      if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		std::cout<<"eigen vector "<< iWave<<": "<<residualNorm[iWave]<<std::endl;
	    }
	}

      if(dftParameters::verbosity>=4)
      {
	if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	  std::cout <<std::endl;
      }

    }
#endif

#ifdef USE_COMPLEX
    unsigned int lowdenOrthogonalization(dealii::parallel::distributed::Vector<std::complex<double> > & X,
				 const unsigned int numberVectors)
    {
      const unsigned int localVectorSize = X.local_size()/numberVectors;
      std::vector<std::complex<double> > overlapMatrix(numberVectors*numberVectors,0.0);

      //
      //blas level 3 dgemm flags
      //
      const double alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;

      //
      //compute overlap matrix S = {(Zc)^T}*Z on local proc
      //where Z is a matrix with size number of degrees of freedom times number of column vectors
      //and (Zc)^T is conjugate transpose of Z
      //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
      //corresponding to column-major format required for blas, we compute
      //the transpose of overlap matrix i.e S^{T} = X*{(Xc)^T} here
      //
      const char uplo = 'U';
      const char trans = 'N';

      zherk_(&uplo,
	     &trans,
	     &numberVectors,
	     &localVectorSize,
	     &alpha,
	     X.begin(),
	     &numberVectors,
	     &beta,
	     &overlapMatrix[0],
	     &numberVectors);


      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix);

      //
      //evaluate the conjugate of {S^T} to get actual overlap matrix
      //
      for(unsigned int i = 0; i < overlapMatrix.size(); ++i)
	overlapMatrix[i] = std::conj(overlapMatrix[i]);


      //
      //set lapack eigen decomposition flags and compute eigendecomposition of S = Q*D*Q^{H}
      //
      int info;
      const unsigned int lwork = 1 + 6*numberVectors + 2*numberVectors*numberVectors, liwork = 3 + 5*numberVectors;
      std::vector<int> iwork(liwork,0);
      const char jobz='V';
      const unsigned int lrwork = 1 + 5*numberVectors + 2*numberVectors*numberVectors;
      std::vector<double> rwork(lrwork,0.0);
      std::vector<std::complex<double> > work(lwork);
      std::vector<double> eigenValuesOverlap(numberVectors,0.0);

      zheevd_(&jobz,
	      &uplo,
	      &numberVectors,
	      &overlapMatrix[0],
	      &numberVectors,
	      &eigenValuesOverlap[0],
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);

       //
       //free up memory associated with work
       //
       work.clear();
       iwork.clear();
       rwork.clear();
       std::vector<std::complex<double> >().swap(work);
       std::vector<double>().swap(rwork);
       std::vector<int>().swap(iwork);

       //
       //compute D^{-1/4} where S = Q*D*Q^{H}
       //
       std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues,0.0);

       unsigned int nanFlag = 0;
       for(unsigned i = 0; i < numberEigenValues; ++i)
	{
	  invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],1.0/4);
	  if(std::isnan(invFourthRootEigenValuesMatrix[i]) || eigenValuesOverlap[i]<1e-13)
	    {
	      nanFlag = 1;
	      break;
	    }
	}
       nanFlag=dealii::Utilities::MPI::max(nanFlag,X.get_mpi_communicator());
       if (dftParameters::enableSwitchToGS && nanFlag==1)
          return nanFlag;

       //
       //Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling "zheevd"
       //
       const unsigned int inc = 1;
       for(unsigned int i = 0; i < numberEigenValues; ++i)
	 {
	   const double scalingCoeff = invFourthRootEigenValuesMatrix[i];
	   zdscal_(&numberEigenValues,
		  &scalingCoeff,
		  &overlapMatrix[0]+i*numberEigenValues,
                  &inc);
	 }

       //
       //Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{H} = (Q*D^{-1/4})*(Q*D^{-1/4))^{H}
       //
       std::vector<std::complex<double> > invSqrtOverlapMatrix(numberEigenValues*numberEigenValues,0.0);
       const char transA1 = 'N';
       const char transB1 = 'C';
       const std::complex<double> alpha1 = 1.0, beta1 = 0.0;


       zgemm_(&transA1,
	      &transB1,
	      &numberEigenValues,
	      &numberEigenValues,
	      &numberEigenValues,
	      &alpha1,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &beta1,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues);

       //
       //free up memory associated with overlapMatrix
       //
       overlapMatrix.clear();
       std::vector<std::complex<double> >().swap(overlapMatrix);

       //
       //Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but implemented as Y^T = {S^{-1/2}}^T*{X^T}
       //using the column major format of blas
       //
       const char transA2  = 'T', transB2  = 'N';
       dealii::parallel::distributed::Vector<std::complex<double> > orthoNormalizedBasis;
       orthoNormalizedBasis.reinit(X);
       zgemm_(&transA2,
	     &transB2,
	     &numberEigenValues,
             &localVectorSize,
	     &numberEigenValues,
	     &alpha1,
	     &invSqrtOverlapMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta1,
	     orthoNormalizedBasis.begin(),
	     &numberEigenValues);


       X = orthoNormalizedBasis;

       return 0;
    }
#else
    unsigned int lowdenOrthogonalization(dealii::parallel::distributed::Vector<double> & X,
				 const unsigned int numberVectors)
    {
      const unsigned int localVectorSize = X.local_size()/numberVectors;

      std::vector<double> overlapMatrix(numberVectors*numberVectors,0.0);


      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ||
					  dftParameters::verbosity<4? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);




      //
      //blas level 3 dgemm flags
      //
      const double alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;
      const char uplo = 'U';
      const char trans = 'N';

      //
      //compute overlap matrix S = {(Z)^T}*Z on local proc
      //where Z is a matrix with size number of degrees of freedom times number of column vectors
      //and (Z)^T is transpose of Z
      //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
      //corresponding to column-major format required for blas, we compute
      //the overlap matrix as S = S^{T} = X*{X^T} here
      //

      computing_timer.enter_section("local overlap matrix for lowden");
      dsyrk_(&uplo,
	     &trans,
	     &numberVectors,
	     &localVectorSize,
	     &alpha,
	     X.begin(),
	     &numberVectors,
	     &beta,
	     &overlapMatrix[0],
	     &numberVectors);
      computing_timer.exit_section("local overlap matrix for lowden");

      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix);

      std::vector<double> eigenValuesOverlap(numberVectors);
      computing_timer.enter_section("eigen decomp. of overlap matrix");
      callevd(numberVectors,
	      &overlapMatrix[0],
	      &eigenValuesOverlap[0]);
      computing_timer.exit_section("eigen decomp. of overlap matrix");

      //
      //compute D^{-1/4} where S = Q*D*Q^{T}
      //
      std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues);
      unsigned int nanFlag = 0;
      for(unsigned i = 0; i < numberEigenValues; ++i)
	{
	  invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],1.0/4);
	  if(std::isnan(invFourthRootEigenValuesMatrix[i]) || eigenValuesOverlap[i]<1e-10)
	    {
	      nanFlag = 1;
	      break;
	    }
	}

      nanFlag=dealii::Utilities::MPI::max(nanFlag,X.get_mpi_communicator());
      if (dftParameters::enableSwitchToGS && nanFlag==1)
          return nanFlag;

      if(nanFlag == 1)
	{
	  std::cout<<"Nan obtained: switching to more robust dsyevr for eigen decomposition "<<std::endl;
	  std::vector<double> overlapMatrixEigenVectors(numberVectors*numberVectors,0.0);
	  eigenValuesOverlap.clear();
	  eigenValuesOverlap.resize(numberVectors);
	  invFourthRootEigenValuesMatrix.clear();
	  invFourthRootEigenValuesMatrix.resize(numberVectors);
	  computing_timer.enter_section("eigen decomp. of overlap matrix");
	  callevr(numberVectors,
		  &overlapMatrix[0],
		  &overlapMatrixEigenVectors[0],
		  &eigenValuesOverlap[0]);
	  computing_timer.exit_section("eigen decomp. of overlap matrix");

	  overlapMatrix = overlapMatrixEigenVectors;
	  overlapMatrixEigenVectors.clear();
	  std::vector<double>().swap(overlapMatrixEigenVectors);

	  //
	  //compute D^{-1/4} where S = Q*D*Q^{T}
	  //
	  for(unsigned i = 0; i < numberEigenValues; ++i)
	    {
	      invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],(1.0/4.0));
	      AssertThrow(!std::isnan(invFourthRootEigenValuesMatrix[i]),dealii::ExcMessage("Eigen values of overlap matrix during Lowden Orthonormalization are close to zero."));
	    }
	}

       //
       //Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling "dsyevd"
       //
      computing_timer.enter_section("scaling in Lowden");
      const unsigned int inc = 1;
      for(unsigned int i = 0; i < numberEigenValues; ++i)
	{
	  double scalingCoeff = invFourthRootEigenValuesMatrix[i];
	  dscal_(&numberEigenValues,
		 &scalingCoeff,
		 &overlapMatrix[0]+i*numberEigenValues,
		 &inc);
	}
      computing_timer.exit_section("scaling in Lowden");

       //
       //Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{T} = (Q*D^{-1/4})*(Q*D^{-1/4}))^{T}
       //
       std::vector<double> invSqrtOverlapMatrix(numberEigenValues*numberEigenValues,0.0);
       const char transA1 = 'N';
       const char transB1 = 'T';
       computing_timer.enter_section("inverse sqrt overlap");
       dgemm_(&transA1,
	      &transB1,
	      &numberEigenValues,
	      &numberEigenValues,
	      &numberEigenValues,
	      &alpha,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &beta,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues);
       computing_timer.exit_section("inverse sqrt overlap");

       //
       //free up memory associated with overlapMatrix
       //
       overlapMatrix.clear();
       std::vector<double>().swap(overlapMatrix);

       //
       //Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but implemented as Yt = S^{-1/2}*Xt
       //using the column major format of blas
       //
       const char transA2  = 'N', transB2  = 'N';
       dealii::parallel::distributed::Vector<double> orthoNormalizedBasis;
       orthoNormalizedBasis.reinit(X);
       computing_timer.enter_section("subspace rotation in lowden");
       dgemm_(&transA2,
	      &transB2,
	      &numberEigenValues,
	      &localVectorSize,
	      &numberEigenValues,
	      &alpha,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues,
	      X.begin(),
	      &numberEigenValues,
	      &beta,
	      orthoNormalizedBasis.begin(),
	      &numberEigenValues);
       computing_timer.exit_section("subspace rotation in lowden");


       X = orthoNormalizedBasis;

       return 0;
    }
#endif



    template void chebyshevFilter(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<dataTypes::number> & ,
				  const unsigned int ,
				  const unsigned int,
				  const double ,
				  const double ,
				  const double );


    template void gramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<dataTypes::number> &,
					       const unsigned int);

    template unsigned int pseudoGramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<dataTypes::number> &,
					             const unsigned int,
						     const MPI_Comm &);

    template void rayleighRitz(operatorDFTClass  & operatorMatrix,
			       dealii::parallel::distributed::Vector<dataTypes::number> &,
			       const unsigned int numberWaveFunctions,
			       const MPI_Comm &,
			       std::vector<double>     & eigenValues);

    template void computeEigenResidualNorm(operatorDFTClass        & operatorMatrix,
					   dealii::parallel::distributed::Vector<dataTypes::number> & X,
					   const std::vector<double> & eigenValues,
					   std::vector<double>     & residualNorm);

  }//end of namespace

}
