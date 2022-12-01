// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh, Sambit Das
//

// source file for electron density related computations

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::popOutRhoInRhoOutVals()
{
  // pop out rhoInVals and rhoOutVals if their size exceeds mixing history size

  if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
      d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
    {
      if (d_rhoInNodalVals.size() == d_dftParamsPtr->mixingHistory)
        {
          d_rhoInNodalVals.pop_front();
          d_rhoOutNodalVals.pop_front();

          if (d_dftParamsPtr->spinPolarized == 1)
            {
              d_rhoInSpin0NodalVals.pop_front();
              d_rhoOutSpin0NodalVals.pop_front();
              d_rhoInSpin1NodalVals.pop_front();
              d_rhoOutSpin1NodalVals.pop_front();
            }
        }
    }
  else
    {
      // pop out rhoInVals and rhoOutVals if their size exceeds mixing history
      // size
      if (rhoInVals.size() == d_dftParamsPtr->mixingHistory)
        {
          rhoInVals.pop_front();
          rhoOutVals.pop_front();

          if (d_dftParamsPtr->spinPolarized == 1)
            {
              rhoInValsSpinPolarized.pop_front();
              rhoOutValsSpinPolarized.pop_front();
            }

          if (excFunctionalPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA) // GGA
            {
              gradRhoInVals.pop_front();
              gradRhoOutVals.pop_front();
            }

          if (d_dftParamsPtr->spinPolarized == 1 &&
              excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
            {
              gradRhoInValsSpinPolarized.pop_front();
              gradRhoOutValsSpinPolarized.pop_front();
            }

          if (d_dftParamsPtr->mixingMethod == "BROYDEN")
            {
              dFBroyden.pop_front();
              uBroyden.pop_front();
              if (excFunctionalPtr->getDensityBasedFamilyType() ==
                  densityFamilyType::GGA) // GGA
                {
                  graddFBroyden.pop_front();
                  gradUBroyden.pop_front();
                }
            }
        }
    }
}


// calculate electron density
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::compute_rhoOut(
#ifdef DFTFE_WITH_DEVICE
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
    &kohnShamDFTEigenOperator,
#endif
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
    &        kohnShamDFTEigenOperatorCPU,
  const bool isConsiderSpectrumSplitting,
  const bool isGroundState)
{
  if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
      d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
    {
#ifdef DFTFE_WITH_DEVICE
      computeRhoNodalFromPSI(kohnShamDFTEigenOperator,
                             kohnShamDFTEigenOperatorCPU,
                             isConsiderSpectrumSplitting);
#else
      computeRhoNodalFromPSI(kohnShamDFTEigenOperatorCPU,
                             isConsiderSpectrumSplitting);
#endif
      d_rhoOutNodalValues.update_ghost_values();

      // normalize rho
      const double charge =
        totalCharge(d_matrixFreeDataPRefined, d_rhoOutNodalValues);


      const double scalingFactor = ((double)numElectrons) / charge;

      // scale nodal vector with scalingFactor
      d_rhoOutNodalValues *= scalingFactor;

      d_rhoOutNodalVals.push_back(d_rhoOutNodalValues);


      // interpolate nodal rhoOut data to quadrature data
      interpolateRhoNodalDataToQuadratureDataGeneral(
        d_matrixFreeDataPRefined,
        d_densityDofHandlerIndexElectro,
        d_densityQuadratureIdElectro,
        d_rhoOutNodalValues,
        *rhoOutValues,
        *gradRhoOutValues,
        *gradRhoOutValues,
        excFunctionalPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA);

      if (d_dftParamsPtr->spinPolarized == 1)
        {
          d_rhoOutSpin0NodalValues.update_ghost_values();
          d_rhoOutSpin0NodalValues *= scalingFactor;
          d_rhoOutSpin0NodalVals.push_back(d_rhoOutSpin0NodalValues);


          d_rhoOutSpin1NodalValues.update_ghost_values();
          d_rhoOutSpin1NodalValues *= scalingFactor;
          d_rhoOutSpin1NodalVals.push_back(d_rhoOutSpin1NodalValues);

          // interpolate nodal rhoOut data to quadrature data
          interpolateRhoSpinNodalDataToQuadratureDataGeneral(
            d_matrixFreeDataPRefined,
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_rhoOutSpin0NodalValues,
            d_rhoOutSpin1NodalValues,
            *rhoOutValuesSpinPolarized,
            *gradRhoOutValuesSpinPolarized,
            *gradRhoOutValuesSpinPolarized,
            excFunctionalPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA);
        }


      if (d_dftParamsPtr->verbosity >= 3)
        {
          pcout << "Total Charge using nodal Rho out: "
                << totalCharge(d_matrixFreeDataPRefined, d_rhoOutNodalValues)
                << std::endl;
        }
    }
  else
    {
      resizeAndAllocateRhoTableStorage(rhoOutVals,
                                       gradRhoOutVals,
                                       rhoOutValsSpinPolarized,
                                       gradRhoOutValsSpinPolarized);

      rhoOutValues = &(rhoOutVals.back());
      if (d_dftParamsPtr->spinPolarized == 1)
        rhoOutValuesSpinPolarized = &(rhoOutValsSpinPolarized.back());

      if (excFunctionalPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA)
        {
          gradRhoOutValues = &(gradRhoOutVals.back());
          if (d_dftParamsPtr->spinPolarized == 1)
            gradRhoOutValuesSpinPolarized =
              &(gradRhoOutValsSpinPolarized.back());
        }

#ifdef DFTFE_WITH_DEVICE
      if (d_dftParamsPtr->useDevice)
        Device::computeRhoFromPSI(
          d_eigenVectorsFlattenedDevice.begin(),
          d_eigenVectorsRotFracFlattenedDevice.begin(),
          d_numEigenValues,
          d_numEigenValuesRR,
          d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
          eigenValues,
          fermiEnergy,
          fermiEnergyUp,
          fermiEnergyDown,
          kohnShamDFTEigenOperator,
          d_eigenDofHandlerIndex,
          dofHandler,
          matrix_free_data.n_physical_cells(),
          matrix_free_data.get_dofs_per_cell(d_densityDofHandlerIndex),
          matrix_free_data.get_quadrature(d_densityQuadratureId).size(),
          d_kPointWeights,
          rhoOutValues,
          gradRhoOutValues,
          rhoOutValuesSpinPolarized,
          gradRhoOutValuesSpinPolarized,
          excFunctionalPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA,
          d_mpiCommParent,
          interpoolcomm,
          interBandGroupComm,
          *d_dftParamsPtr,
          isConsiderSpectrumSplitting &&
            d_numEigenValues != d_numEigenValuesRR);
#endif
      if (!d_dftParamsPtr->useDevice)
        computeRhoFromPSICPU(
          d_eigenVectorsFlattenedSTL,
          d_eigenVectorsRotFracDensityFlattenedSTL,
          d_numEigenValues,
          d_numEigenValuesRR,
          d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
          eigenValues,
          fermiEnergy,
          fermiEnergyUp,
          fermiEnergyDown,
          kohnShamDFTEigenOperatorCPU,
          dofHandler,
          matrix_free_data.n_physical_cells(),
          matrix_free_data.get_dofs_per_cell(d_densityDofHandlerIndex),
          matrix_free_data.get_quadrature(d_densityQuadratureId).size(),
          d_kPointWeights,
          rhoOutValues,
          gradRhoOutValues,
          rhoOutValuesSpinPolarized,
          gradRhoOutValuesSpinPolarized,
          excFunctionalPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA,
          d_mpiCommParent,
          interpoolcomm,
          interBandGroupComm,
          *d_dftParamsPtr,
          isConsiderSpectrumSplitting && d_numEigenValues != d_numEigenValuesRR,
          false);
      // normalizeRhoOutQuadValues();

      if (isGroundState)
        {
          computeRhoNodalFromPSI(
#ifdef DFTFE_WITH_DEVICE
            kohnShamDFTEigenOperator,
#endif
            kohnShamDFTEigenOperatorCPU,
            isConsiderSpectrumSplitting);
          d_rhoOutNodalValues.update_ghost_values();

          // normalize rho
          const double charge =
            totalCharge(d_matrixFreeDataPRefined, d_rhoOutNodalValues);


          const double scalingFactor = ((double)numElectrons) / charge;

          // scale nodal vector with scalingFactor
          d_rhoOutNodalValues *= scalingFactor;
        }
    }

  if (isGroundState)
    {
      d_rhoOutNodalValuesDistributed = d_rhoOutNodalValues;
      d_constraintsRhoNodalInfo.distribute(d_rhoOutNodalValuesDistributed);
      interpolateRhoNodalDataToQuadratureDataLpsp(
        d_matrixFreeDataPRefined,
        d_densityDofHandlerIndexElectro,
        d_lpspQuadratureIdElectro,
        d_rhoOutNodalValues,
        d_rhoOutValuesLpspQuad,
        d_gradRhoOutValuesLpspQuad,
        true);
    }
  else if (d_dftParamsPtr->computeEnergyEverySCF)
    {
      if (d_dftParamsPtr->mixingMethod != "ANDERSON_WITH_KERKER" &&
          d_dftParamsPtr->mixingMethod != "LOW_RANK_DIELECM_PRECOND")
        {
          std::function<double(
            const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const unsigned int                                          q)>
            funcRho =
              [&](const typename dealii::DoFHandler<3>::active_cell_iterator
                    &                cell,
                  const unsigned int q) {
                return (*rhoOutValues).find(cell->id())->second[q];
              };
          dealii::VectorTools::project<3, distributedCPUVec<double>>(
            dealii::MappingQ1<3, 3>(),
            d_dofHandlerRhoNodal,
            d_constraintsRhoNodal,
            d_matrixFreeDataPRefined.get_quadrature(
              d_densityQuadratureIdElectro),
            funcRho,
            d_rhoOutNodalValues);
          d_rhoOutNodalValues.update_ghost_values();
        }

      interpolateRhoNodalDataToQuadratureDataLpsp(
        d_matrixFreeDataPRefined,
        d_densityDofHandlerIndexElectro,
        d_lpspQuadratureIdElectro,
        d_rhoOutNodalValues,
        d_rhoOutValuesLpspQuad,
        d_gradRhoOutValuesLpspQuad,
        true);
    }

  popOutRhoInRhoOutVals();

  if (isGroundState &&
      ((d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
        d_dftParamsPtr->solverMode == "GEOOPT") ||
       (d_dftParamsPtr->extrapolateDensity == 2 &&
        d_dftParamsPtr->solverMode == "MD")) &&
      d_dftParamsPtr->spinPolarized != 1)
    {
      d_rhoOutNodalValuesSplit = d_rhoOutNodalValues;
      std::map<dealii::CellId, std::vector<double>> rhoOutValuesCopy =
        *(rhoOutValues);

      const Quadrature<3> &quadrature_formula =
        matrix_free_data.get_quadrature(d_densityQuadratureId);
      const unsigned int n_q_points = quadrature_formula.size();

      const double charge  = totalCharge(d_dofHandlerRhoNodal, rhoOutValues);
      const double scaling = ((double)numElectrons) / charge;

      // scaling rho
      typename DoFHandler<3>::active_cell_iterator cell =
                                                     dofHandler.begin_active(),
                                                   endc = dofHandler.end();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  rhoOutValuesCopy[cell->id()][q] *= scaling;
                }
            }
        }
      l2ProjectionQuadDensityMinusAtomicDensity(d_matrixFreeDataPRefined,
                                                d_constraintsRhoNodal,
                                                d_densityDofHandlerIndexElectro,
                                                d_densityQuadratureIdElectro,
                                                rhoOutValuesCopy,
                                                d_rhoOutNodalValuesSplit);
      d_rhoOutNodalValuesSplit.update_ghost_values();
    }
}



template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::resizeAndAllocateRhoTableStorage(
  std::deque<std::map<dealii::CellId, std::vector<double>>> &rhoVals,
  std::deque<std::map<dealii::CellId, std::vector<double>>> &gradRhoVals,
  std::deque<std::map<dealii::CellId, std::vector<double>>>
    &rhoValsSpinPolarized,
  std::deque<std::map<dealii::CellId, std::vector<double>>>
    &gradRhoValsSpinPolarized)
{
  const unsigned int numQuadPoints =
    matrix_free_data.get_n_q_points(d_densityQuadratureId);
  ;

  // create new rhoValue tables
  rhoVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  if (d_dftParamsPtr->spinPolarized == 1)
    rhoValsSpinPolarized.push_back(
      std::map<dealii::CellId, std::vector<double>>());

  if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
    {
      gradRhoVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      if (d_dftParamsPtr->spinPolarized == 1)
        gradRhoValsSpinPolarized.push_back(
          std::map<dealii::CellId, std::vector<double>>());
    }


  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        const dealii::CellId cellId = cell->id();
        rhoVals.back()[cellId]      = std::vector<double>(numQuadPoints, 0.0);
        if (excFunctionalPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          gradRhoVals.back()[cellId] =
            std::vector<double>(3 * numQuadPoints, 0.0);

        if (d_dftParamsPtr->spinPolarized == 1)
          {
            rhoValsSpinPolarized.back()[cellId] =
              std::vector<double>(2 * numQuadPoints, 0.0);
            if (excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              gradRhoValsSpinPolarized.back()[cellId] =
                std::vector<double>(6 * numQuadPoints, 0.0);
          }
      }
}


// rho data reinitilization without remeshing. The rho out of last ground state
// solve is made the rho in of the new solve
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::noRemeshRhoDataInit()
{
  if (rhoOutVals.size() > 0 || d_rhoInNodalVals.size() > 0)
    {
      // create temporary copies of rho Out data
      std::map<dealii::CellId, std::vector<double>> rhoOutValuesCopy =
        *(rhoOutValues);

      std::map<dealii::CellId, std::vector<double>> gradRhoOutValuesCopy;
      if (excFunctionalPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA)
        {
          gradRhoOutValuesCopy = *(gradRhoOutValues);
        }

      std::map<dealii::CellId, std::vector<double>>
        rhoOutValuesSpinPolarizedCopy;
      if (d_dftParamsPtr->spinPolarized == 1)
        {
          rhoOutValuesSpinPolarizedCopy = *(rhoOutValuesSpinPolarized);
        }

      std::map<dealii::CellId, std::vector<double>>
        gradRhoOutValuesSpinPolarizedCopy;
      if (d_dftParamsPtr->spinPolarized == 1 &&
          excFunctionalPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
        {
          gradRhoOutValuesSpinPolarizedCopy = *(gradRhoOutValuesSpinPolarized);
        }

      // cleanup of existing rho Out and rho In data
      clearRhoData();

      /// copy back temporary rho out to rho in data
      rhoInVals.push_back(rhoOutValuesCopy);
      rhoInValues = &(rhoInVals.back());

      if (excFunctionalPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA)
        {
          gradRhoInVals.push_back(gradRhoOutValuesCopy);
          gradRhoInValues = &(gradRhoInVals.back());
        }

      if (d_dftParamsPtr->spinPolarized == 1)
        {
          rhoInValsSpinPolarized.push_back(rhoOutValuesSpinPolarizedCopy);
          rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());
        }

      if (excFunctionalPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA &&
          d_dftParamsPtr->spinPolarized == 1)
        {
          gradRhoInValsSpinPolarized.push_back(
            gradRhoOutValuesSpinPolarizedCopy);
          gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
        }

      if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
          d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
        {
          d_rhoInNodalValues = d_rhoOutNodalValues;
          d_rhoInNodalValues.update_ghost_values();

          // normalize rho
          const double charge =
            totalCharge(d_matrixFreeDataPRefined, d_rhoInNodalValues);

          const double scalingFactor = ((double)numElectrons) / charge;

          // scale nodal vector with scalingFactor
          d_rhoInNodalValues *= scalingFactor;
          d_rhoInNodalVals.push_back(d_rhoInNodalValues);

          interpolateRhoNodalDataToQuadratureDataGeneral(
            d_matrixFreeDataPRefined,
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_rhoInNodalValues,
            *rhoInValues,
            *gradRhoInValues,
            *gradRhoInValues,
            excFunctionalPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA);

          if (d_dftParamsPtr->spinPolarized == 1)
            {
              d_rhoInSpin0NodalValues = d_rhoOutSpin0NodalValues;
              d_rhoInSpin0NodalValues.update_ghost_values();

              d_rhoInSpin1NodalValues = d_rhoOutSpin1NodalValues;
              d_rhoInSpin1NodalValues.update_ghost_values();

              // scale nodal vector with scalingFactor
              d_rhoInSpin0NodalValues *= scalingFactor;
              d_rhoInSpin0NodalVals.push_back(d_rhoInSpin0NodalValues);

              d_rhoInSpin1NodalValues *= scalingFactor;
              d_rhoInSpin1NodalVals.push_back(d_rhoInSpin1NodalValues);

              interpolateRhoSpinNodalDataToQuadratureDataGeneral(
                d_matrixFreeDataPRefined,
                d_densityDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_rhoInSpin0NodalValues,
                d_rhoInSpin1NodalValues,
                *rhoInValuesSpinPolarized,
                *gradRhoInValuesSpinPolarized,
                *gradRhoInValuesSpinPolarized,
                excFunctionalPtr->getDensityBasedFamilyType() ==
                  densityFamilyType::GGA);
            }

          rhoOutVals.push_back(std::map<dealii::CellId, std::vector<double>>());
          rhoOutValues = &(rhoOutVals.back());
          if (d_dftParamsPtr->spinPolarized == 1)
            {
              rhoOutValsSpinPolarized.push_back(
                std::map<dealii::CellId, std::vector<double>>());
              rhoOutValuesSpinPolarized = &(rhoOutValsSpinPolarized.back());
            }

          if (excFunctionalPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA)
            {
              gradRhoOutVals.push_back(
                std::map<dealii::CellId, std::vector<double>>());
              gradRhoOutValues = &(gradRhoOutVals.back());

              if (d_dftParamsPtr->spinPolarized == 1)
                {
                  gradRhoOutValsSpinPolarized.push_back(
                    std::map<dealii::CellId, std::vector<double>>());
                  gradRhoOutValuesSpinPolarized =
                    &(gradRhoOutValsSpinPolarized.back());
                }
            }
        }

      // scale quadrature values
      normalizeRhoInQuadValues();
    }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::computeRhoNodalFromPSI(
#ifdef DFTFE_WITH_DEVICE
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
    &kohnShamDFTEigenOperator,
#endif
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
    &  kohnShamDFTEigenOperatorCPU,
  bool isConsiderSpectrumSplitting)
{
  std::map<dealii::CellId, std::vector<double>> rhoPRefinedNodalData;
  std::map<dealii::CellId, std::vector<double>>
    rhoPRefinedSpinPolarizedNodalData;

  // initialize variables to be used later
  const unsigned int dofs_per_cell =
    d_dofHandlerRhoNodal.get_fe().dofs_per_cell;
  typename DoFHandler<3>::active_cell_iterator cell = d_dofHandlerRhoNodal
                                                        .begin_active(),
                                               endc =
                                                 d_dofHandlerRhoNodal.end();
  const dealii::IndexSet &locallyOwnedDofs =
    d_dofHandlerRhoNodal.locally_owned_dofs();
  const Quadrature<3> &quadrature_formula =
    matrix_free_data.get_quadrature(d_gllQuadratureId);
  const unsigned int numQuadPoints = quadrature_formula.size();

  // get access to quadrature point coordinates and 2p DoFHandler nodal points
  const std::vector<Point<3>> &quadraturePointCoor =
    quadrature_formula.get_points();
  const std::vector<Point<3>> &supportPointNaturalCoor =
    d_dofHandlerRhoNodal.get_fe().get_unit_support_points();
  std::vector<unsigned int> renumberingMap(numQuadPoints);

  // create renumbering map between the numbering order of quadrature points and
  // lobatto support points
  for (unsigned int i = 0; i < numQuadPoints; ++i)
    {
      const Point<3> &nodalCoor = supportPointNaturalCoor[i];
      for (unsigned int j = 0; j < numQuadPoints; ++j)
        {
          const Point<3> &quadCoor = quadraturePointCoor[j];
          double          dist     = quadCoor.distance(nodalCoor);
          if (dist <= 1e-08)
            {
              renumberingMap[i] = j;
              break;
            }
        }
    }

  // allocate the storage to compute 2p nodal values from wavefunctions
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellId = cell->id();
          rhoPRefinedNodalData[cellId] =
            std::vector<double>(numQuadPoints, 0.0);

          if (d_dftParamsPtr->spinPolarized == 1)
            {
              rhoPRefinedSpinPolarizedNodalData[cellId] =
                std::vector<double>(numQuadPoints * 2, 0.0);
            }
        }
    }

  // allocate dummy datastructures
  std::map<dealii::CellId, std::vector<double>> _gradRhoValues;
  std::map<dealii::CellId, std::vector<double>> _gradRhoValuesSpinPolarized;

  cell = dofHandler.begin_active();
  endc = dofHandler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        const dealii::CellId cellId = cell->id();
        if (excFunctionalPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          (_gradRhoValues)[cellId] =
            std::vector<double>(3 * numQuadPoints, 0.0);

        if (d_dftParamsPtr->spinPolarized == 1)
          {
            if (excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              (_gradRhoValuesSpinPolarized)[cellId] =
                std::vector<double>(6 * numQuadPoints, 0.0);
          }
      }


      // compute rho from wavefunctions at nodal locations of 2p DoFHandler
      // nodes in each cell
#ifdef DFTFE_WITH_DEVICE
  if (d_dftParamsPtr->useDevice)
    Device::computeRhoFromPSI(
      d_eigenVectorsFlattenedDevice.begin(),
      d_eigenVectorsRotFracFlattenedDevice.begin(),
      d_numEigenValues,
      d_numEigenValuesRR,
      d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
      eigenValues,
      fermiEnergy,
      fermiEnergyUp,
      fermiEnergyDown,
      kohnShamDFTEigenOperator,
      d_eigenDofHandlerIndex,
      dofHandler,
      matrix_free_data.n_physical_cells(),
      matrix_free_data.get_dofs_per_cell(d_densityDofHandlerIndex),
      quadrature_formula.size(),
      d_kPointWeights,
      &rhoPRefinedNodalData,
      &_gradRhoValues,
      &rhoPRefinedSpinPolarizedNodalData,
      &_gradRhoValuesSpinPolarized,
      false,
      d_mpiCommParent,
      interpoolcomm,
      interBandGroupComm,
      *d_dftParamsPtr,
      isConsiderSpectrumSplitting && d_numEigenValues != d_numEigenValuesRR,
      true);
#endif
  if (!d_dftParamsPtr->useDevice)
    computeRhoFromPSICPU(
      d_eigenVectorsFlattenedSTL,
      d_eigenVectorsRotFracDensityFlattenedSTL,
      d_numEigenValues,
      d_numEigenValuesRR,
      d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
      eigenValues,
      fermiEnergy,
      fermiEnergyUp,
      fermiEnergyDown,
      kohnShamDFTEigenOperatorCPU,
      dofHandler,
      matrix_free_data.n_physical_cells(),
      matrix_free_data.get_dofs_per_cell(d_densityDofHandlerIndex),
      quadrature_formula.size(),
      d_kPointWeights,
      &rhoPRefinedNodalData,
      &_gradRhoValues,
      &rhoPRefinedSpinPolarizedNodalData,
      &_gradRhoValuesSpinPolarized,
      false,
      d_mpiCommParent,
      interpoolcomm,
      interBandGroupComm,
      *d_dftParamsPtr,
      isConsiderSpectrumSplitting && d_numEigenValues != d_numEigenValuesRR,
      true);

  // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
  DoFHandler<3>::active_cell_iterator cellP =
                                        d_dofHandlerRhoNodal.begin_active(),
                                      endcP = d_dofHandlerRhoNodal.end();

  for (; cellP != endcP; ++cellP)
    {
      if (cellP->is_locally_owned())
        {
          std::vector<dealii::types::global_dof_index> cell_dof_indices(
            dofs_per_cell);
          cellP->get_dof_indices(cell_dof_indices);
          const std::vector<double> &nodalValues =
            rhoPRefinedNodalData.find(cellP->id())->second;
          Assert(
            nodalValues.size() == dofs_per_cell,
            ExcMessage(
              "Number of nodes in 2p DoFHandler does not match with data stored in rhoNodal Values variable"));

          for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
            {
              const dealii::types::global_dof_index nodeID =
                cell_dof_indices[iNode];
              if (!d_constraintsRhoNodal.is_constrained(nodeID))
                {
                  if (locallyOwnedDofs.is_element(nodeID))
                    d_rhoOutNodalValues(nodeID) =
                      nodalValues[renumberingMap[iNode]];
                }
            }
        }
    }

  cellP = d_dofHandlerRhoNodal.begin_active();
  endcP = d_dofHandlerRhoNodal.end();

  if (d_dftParamsPtr->spinPolarized == 1)
    {
      for (; cellP != endcP; ++cellP)
        {
          if (cellP->is_locally_owned())
            {
              std::vector<dealii::types::global_dof_index> cell_dof_indices(
                dofs_per_cell);
              cellP->get_dof_indices(cell_dof_indices);
              const std::vector<double> &nodalValues =
                rhoPRefinedSpinPolarizedNodalData.find(cellP->id())->second;
              Assert(
                nodalValues.size() == 2 * dofs_per_cell,
                ExcMessage(
                  "Number of nodes in 2p DoFHandler does not match with data stored in rhoNodal Values variable"));

              for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
                {
                  const dealii::types::global_dof_index nodeID =
                    cell_dof_indices[iNode];
                  if (!d_constraintsRhoNodal.is_constrained(nodeID))
                    {
                      if (locallyOwnedDofs.is_element(nodeID))
                        {
                          d_rhoOutSpin0NodalValues(nodeID) =
                            nodalValues[2 * renumberingMap[iNode]];
                          d_rhoOutSpin1NodalValues(nodeID) =
                            nodalValues[2 * renumberingMap[iNode] + 1];
                        }
                    }
                }
            }
        }
    }
}
