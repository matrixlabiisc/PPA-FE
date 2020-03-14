// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020  The Regents of the University of Michigan and DFT-FE authors.
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


#include <dealiiLinearSolverProblem.h>

#ifndef poissonSolverProblemCellMatrix_H_
#define poissonSolverProblemCellMatrix_H_

namespace dftfe {

 /**
  * @brief poisson solver problem class template. template parameter FEOrder
  * is the finite element polynomial order
  *
  * @author Sambit Das
  */
  template<unsigned int FEOrder>
  class poissonSolverProblemCellMatrixMultiVector: public dealiiLinearSolverProblem {

    public:

	/// Constructor
	poissonSolverProblemCellMatrixMultiVector(const  MPI_Comm &mpi_comm);


	/**
	 * @brief reinitialize data structures for nuclear electrostatic potential solve
	 *
	 */
	 void reinit(const dealii::MatrixFree<3,double> & matrixFreeData,
		     vectorType & x,
		     const dealii::ConstraintMatrix & constraintMatrix,
                     const dealii::ConstraintMatrix & constraintMatrixRhs,
		     const unsigned int matrixFreeVectorComponent,
	             const std::map<dealii::types::global_dof_index, double> & atoms,
                     const double * inhomoIdsColoredVecFlattened,
                     const unsigned int numberBins,
                     const unsigned int binId,
		     const bool isComputeDiagonalA=true,
                     const bool isPrecomputeShapeGradIntegral=false);

	/**
	 * @brief get the reference to x field
	 *
	 * @return reference to x field. Assumes x field data structure is already initialized
	 */
	vectorType & getX();

	/**
	 * @brief Compute A matrix multipled by x.
	 *
	 */
	void vmult(vectorType &Ax,
		   const vectorType &x) const;

	/**
	 * @brief Compute right hand side vector for the problem Ax = rhs.
	 *
	 * @param rhs vector for the right hand side values
	 */
	void computeRhs(vectorType & rhs);

	/**
	 * @brief Jacobi preconditioning.
	 *
	 */
        void precondition_Jacobi(vectorType& dst,
		                 const vectorType& src,
				 const double omega) const;

	/**
	 * @brief distribute x to the constrained nodes.
	 *
	 */
	void distributeX();

	/// function needed by dealii to mimic SparseMatrix for Jacobi preconditioning
        void subscribe (std::atomic< bool > *const validity, const std::string &identifier="") const{};

	/// function needed by dealii to mimic SparseMatrix for Jacobi preconditioning
        void unsubscribe (std::atomic< bool > *const validity, const std::string &identifier="") const{};

	/// function needed by dealii to mimic SparseMatrix
        bool operator!= (double val) const {return true;};

    private:

       /**
	 * @brief required for the cell_loop operation in dealii's MatrixFree class
	 *
	 */
        void AX (const dealii::MatrixFree<3,double>  &matrixFreeData,
	                  vectorType &dst,
		          const vectorType &src,
		          const std::pair<unsigned int,unsigned int> &cell_range) const;


	/**
	 * @brief Compute the diagonal of A.
	 *
	 */
	void computeDiagonalA();

	/**
	 * @brief precompute shape function gradient integral.
	 *
	 */
	void precomputeShapeFunctionGradientIntegral();


	/// storage for diagonal of the A matrix
	vectorType d_diagonalA;

	/// pointer to dealii MatrixFree object
        const dealii::MatrixFree<3,double>  * d_matrixFreeDataPtr;

	/// pointer to the x vector being solved for
        vectorType * d_xPtr;

	/// pointer to dealii ConstraintMatrix object
        const dealii::ConstraintMatrix * d_constraintMatrixPtr;

        /// pointer to dealii ConstraintMatrix object
        const dealii::ConstraintMatrix * d_constraintMatrixPtrRhs;

        /// pointer to 
        const double * d_inhomoIdsColoredVecFlattened;

	/// matrix free index required to access the DofHandler and ConstraintMatrix objects corresponding to the
	/// problem
        unsigned int d_matrixFreeVectorComponent;

	/// pointer to map between global dof index in current processor and the atomic charge on that dof
	const std::map<dealii::types::global_dof_index, double> * d_atomsPtr;

        /// shape function gradient integral storage
        std::vector<double> d_cellShapeFunctionGradientIntegralFlattened;

        bool d_isShapeGradIntegralPrecomputed;

        unsigned int d_numberBins;

        unsigned int d_binId;

        const MPI_Comm mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;
        dealii::ConditionalOStream   pcout;
  };

}
#endif // poissonSolverProblemCellMatrix_H_
