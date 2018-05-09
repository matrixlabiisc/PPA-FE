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



#ifndef NonLinearSolver_h
#define NonLinearSolver_h


#include "headers.h"

namespace dftfe {
  //
  // forward declarations
  //
  class nonlinearSolverProblem;

  /**
   * @brief Base class for non-linear algebraic solver.
   *
   * @author Sambit Das
   */
  class nonLinearSolver {

    //
    // types
    //
  public:
    enum ReturnValueType { SUCCESS = 0,
			   FAILURE,
			   LINESEARCH_FAILED,
			   MAX_ITER_REACHED,
                           RESTART};

    //
    // methods
    //
  public:

    /**
     * @brief Destructor.
     */
    virtual ~nonLinearSolver() = 0;

    /**
     * @brief Solve non-linear algebraic equation.
     *
     * @return Return value indicating success or failure.
     */
     virtual ReturnValueType solve(nonlinearSolverProblem & problem,
	                           const std::string checkpointFileName="") = 0;

    /**
     * @brief Solve non-linear algebraic equation from last saved checkpoint.
     *
     * @return Return value indicating success or failure.
     */
     virtual ReturnValueType restartSolve(nonlinearSolverProblem & problem,
	                                  const std::string checkpointFileName) = 0;

  protected:

    /**
     * @brief Constructor.
     *
     */
    nonLinearSolver(const unsigned int debugLevel,
		    const unsigned int maxNumberIterations,
	            const double tolerance);


  protected:

    /**
     * @brief Get tolerance.
     *
     * @return Value of the tolerance.
     */
    double getTolerance() const;

    /**
     * @brief Get maximum number of iterations.
     *
     * @return Maximum number of iterations.
     */
    unsigned int getMaximumNumberIterations() const;

    /**
     * @brief Get debug level.
     *
     * @return Debug level.
     */
    unsigned int getDebugLevel() const;


    /// controls the verbosity of the printing
    const unsigned int    d_debugLevel;

    /// maximum number of nonlinear solve iterations
    const unsigned int    d_maxNumberIterations;

    /// nonlinear solve stopping tolerance
    const double d_tolerance;

  };

}

#endif // NonLinearSolver_h
