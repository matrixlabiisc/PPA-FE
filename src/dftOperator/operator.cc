
//
// -------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
//
// @author Phani Motamarri
//
#include <operator.h>
#include <linearAlgebraOperationsInternal.h>
#include <dftParameters.h>

//
// Constructor.
//
namespace dftfe {

  operatorDFTClass::operatorDFTClass(const MPI_Comm                                        & mpi_comm_replica,
				     const dealii::MatrixFree<3,double>                    & matrix_free_data,
				     dftUtils::constraintMatrixInfo                        & constraintMatrixNone):
    d_mpi_communicator(mpi_comm_replica),
#ifdef DFTFE_WITH_ELPA	
    d_processGridCommunicatorActive(MPI_COMM_NULL),
    d_processGridCommunicatorActivePartial(MPI_COMM_NULL),
#endif	
    d_matrix_free_data(&matrix_free_data),
    d_constraintMatrixData(&constraintMatrixNone)
  {


  }

  //
  // Destructor.
  //
  operatorDFTClass::~operatorDFTClass()
  {
#ifdef DFTFE_WITH_ELPA	  
      if (d_processGridCommunicatorActive != MPI_COMM_NULL)
	  MPI_Comm_free(&d_processGridCommunicatorActive);

      if (d_processGridCommunicatorActivePartial != MPI_COMM_NULL)
	  MPI_Comm_free(&d_processGridCommunicatorActivePartial);
#endif      
    //
    //
    //
    return;

  }

  //set the data member of operator class
  void operatorDFTClass::setInvSqrtMassVector(distributedCPUVec<double> & invSqrtMassVector) 
  {
    d_invSqrtMassVector = invSqrtMassVector;
  }

  //get access to the data member of operator class
  distributedCPUVec<double> & operatorDFTClass::getInvSqrtMassVector() 
  {
    return d_invSqrtMassVector;
  }

  //
  //Get overloaded constraint matrix object constructed using 1-component FE object
  //
  dftUtils::constraintMatrixInfo * operatorDFTClass::getOverloadedConstraintMatrix() const
  {
    return d_constraintMatrixData;
  }

  //
  //Get matrix free data
  //
  const dealii::MatrixFree<3,double> * operatorDFTClass::getMatrixFreeData() const
  {
    return d_matrix_free_data;
  }

  //
  //Get relevant mpi communicator
  //
  const MPI_Comm & operatorDFTClass::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }


  void operatorDFTClass::processGridOptionalELPASetup(const unsigned int na,
    		                                      const unsigned int nev)
  {


       std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
       linearAlgebraOperations::internal::createProcessGridSquareMatrix(getMPICommunicator(),
                                               na,
                                               processGrid);


       d_scalapackBlockSize=std::min(dftParameters::scalapackBlockSize,
	                     (na+processGrid->get_process_grid_rows()-1)
                             /processGrid->get_process_grid_rows());
#ifdef DFTFE_WITH_ELPA
       if (dftParameters::useELPA)
           linearAlgebraOperations::internal::setupELPAHandle(getMPICommunicator(),
			                                      d_processGridCommunicatorActive,
                                                              processGrid,
							      na,
							      na,
							      d_scalapackBlockSize,
							      d_elpaHandle);
#endif

       if (nev!=na)
       {
#ifdef DFTFE_WITH_ELPA
	   if (dftParameters::useELPA)
	       linearAlgebraOperations::internal::setupELPAHandle(getMPICommunicator(),
			                                          d_processGridCommunicatorActivePartial,
								  processGrid,
								  na,
								  nev,
								  d_scalapackBlockSize,
								  d_elpaHandlePartialEigenVec);
#endif
	   //std::cout<<"nblkvalence: "<<d_scalapackBlockSizeValence<<std::endl;
       }

       //std::cout<<"nblk: "<<d_scalapackBlockSize<<std::endl;

  }

#ifdef DFTFE_WITH_ELPA
  void operatorDFTClass::elpaDeallocateHandles(const unsigned int na,
		                    const unsigned int nev)
  {
       int error;
       elpa_deallocate(d_elpaHandle,&error);
       AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));

       if (na!=nev)
       {

          elpa_deallocate(d_elpaHandlePartialEigenVec,&error);
          AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));
       }
  }
#endif

}
