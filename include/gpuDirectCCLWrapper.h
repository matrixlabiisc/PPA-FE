// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das

#if defined(DFTFE_WITH_GPU)
#ifndef gpuDirectCCLWrapper_h
#define gpuDirectCCLWrapper_h

#include <mpi.h>

namespace dftfe
{
 
  /**
   *  @brief Wrapper class for GPU Direct collective communications library. 
   *
   *  @author Sambit Das
   */
  class GPUCCLWrapper 
  {
      public:

        GPUCCLWrapper();

        void init(const MPI_Comm & mpiComm);

        ~GPUCCLWrapper();

        int gpuDirectAllReduceWrapper(const float *send, float *recv, int size, cudaStream_t & stream);


        int gpuDirectAllReduceWrapper(const double *send, double *recv, int size, cudaStream_t & stream);

        int gpuDirectAllReduceMixedPrecGroupWrapper(const double *send1, const float *send2, double *recv1, float *recv2, int size1, int size2, cudaStream_t & stream);        

      private:

        int myRank;
        int totalRanks;
        bool commCreated;
#ifdef DFTFE_WITH_NCCL       
        void * ncclIdPtr;
        void * ncclCommPtr;
#endif          
  };
}

#endif
#endif
