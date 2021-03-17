// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das
//
// compute configurational stress contribution from all terms except the nuclear
// self energy
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::computeStressEEshelbyEPSPEnlEk(
  const MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_GPU
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
    &kohnShamDFTEigenOperator,
#endif
  const unsigned int               eigenDofHandlerIndex,
  const unsigned int               smearedChargeQuadratureId,
  const unsigned int               lpspQuadratureIdElectro,
  const MatrixFree<3, double> &    matrixFreeDataElectro,
  const unsigned int               phiTotDofHandlerIndexElectro,
  const distributedCPUVec<double> &phiTotRhoOutElectro,
  const std::map<dealii::CellId, std::vector<double>> &rhoOutValues,
  const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValues,
  const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValuesLpsp,
  const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectro,
  const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectroLpsp,
  const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValuesElectro,
  const std::map<dealii::CellId, std::vector<double>>
    &gradRhoOutValuesElectroLpsp,
  const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
  const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
    &                                                  pseudoVLocAtomsElectro,
  const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
  const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
  const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
  const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
    &gradRhoCoreAtoms,
  const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
    &                                              hessianRhoCoreAtoms,
  const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro)
{
  int this_process;
  MPI_Comm_rank(MPI_COMM_WORLD, &this_process);
  MPI_Barrier(MPI_COMM_WORLD);
  double forcetotal_time = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);
  double init_time = MPI_Wtime();

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  std::map<unsigned int, std::vector<double>> forceContributionFnlGammaAtoms;

  const bool isPseudopotential = dftParameters::isPseudopotential;

  FEEvaluation<C_DIM,
               1,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               C_DIM>
    forceEval(matrixFreeData,
              d_forceDofHandlerIndex,
              dftPtr->d_densityQuadratureId);
  FEEvaluation<C_DIM,
               1,
               C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(),
               C_DIM>
    forceEvalNLP(matrixFreeData,
                 d_forceDofHandlerIndex,
                 dftPtr->d_nlpspQuadratureId);
#ifdef USE_COMPLEX
  FEEvaluation<C_DIM,
               1,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               C_DIM>
    forceEvalKPoints(matrixFreeData,
                     d_forceDofHandlerIndex,
                     dftPtr->d_densityQuadratureId);
  FEEvaluation<C_DIM,
               1,
               C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(),
               C_DIM>
    forceEvalKPointsNLP(matrixFreeData,
                        d_forceDofHandlerIndex,
                        dftPtr->d_nlpspQuadratureId);
#endif


#ifdef USE_COMPLEX
  FEEvaluation<C_DIM,
               FEOrder,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               2>
    psiEval(matrixFreeData,
            eigenDofHandlerIndex,
            dftPtr->d_densityQuadratureId);

  FEEvaluation<C_DIM,
               FEOrder,
               C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(),
               2>
    psiEvalNLP(matrixFreeData,
               eigenDofHandlerIndex,
               dftPtr->d_nlpspQuadratureId);
#else
  FEEvaluation<C_DIM,
               FEOrder,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               1>
    psiEval(matrixFreeData,
            eigenDofHandlerIndex,
            dftPtr->d_densityQuadratureId);

  FEEvaluation<C_DIM,
               FEOrder,
               C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(),
               1>
    psiEvalNLP(matrixFreeData,
               eigenDofHandlerIndex,
               dftPtr->d_nlpspQuadratureId);
#endif


  std::map<unsigned int, std::vector<double>>
    forceContributionShadowLocalGammaAtoms;

  const double spinPolarizedFactor =
    (dftParameters::spinPolarized == 1) ? 0.5 : 1.0;
  const VectorizedArray<double> spinPolarizedFactorVect =
    (dftParameters::spinPolarized == 1) ? make_vectorized_array(0.5) :
                                          make_vectorized_array(1.0);

  const unsigned int numQuadPoints    = forceEval.n_q_points;
  const unsigned int numQuadPointsNLP = forceEvalNLP.n_q_points;
  const unsigned int numEigenVectors  = dftPtr->d_numEigenValues;
  const unsigned int numKPoints       = dftPtr->d_kPointWeights.size();

  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
  Tensor<1, 2, VectorizedArray<double>>   zeroTensor1;
  zeroTensor1[0] = make_vectorized_array(0.0);
  zeroTensor1[1] = make_vectorized_array(0.0);
  Tensor<1, 2, Tensor<1, C_DIM, VectorizedArray<double>>> zeroTensor2;
  Tensor<1, C_DIM, VectorizedArray<double>>               zeroTensor3;
  Tensor<2, C_DIM, VectorizedArray<double>>               zeroTensor4;
  Tensor<1, 2, Tensor<2, C_DIM, VectorizedArray<double>>> zeroTensor5;
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    {
      zeroTensor2[0][idim] = make_vectorized_array(0.0);
      zeroTensor2[1][idim] = make_vectorized_array(0.0);
      zeroTensor3[idim]    = make_vectorized_array(0.0);
    }
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    {
      for (unsigned int jdim = 0; jdim < C_DIM; jdim++)
        {
          zeroTensor4[idim][jdim] = make_vectorized_array(0.0);
        }
    }
  zeroTensor5[0] = zeroTensor4;
  zeroTensor5[1] = zeroTensor4;

  std::map<unsigned int, std::vector<unsigned int>>
    macroIdToNonlocalAtomsSetMap;
  for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
    {
      const unsigned int numSubCells = matrixFreeData.n_components_filled(cell);
      std::set<unsigned int> mergedSet;
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId subCellId = subCellPtr->id();

          std::set<unsigned int> s;
          std::set_union(
            mergedSet.begin(),
            mergedSet.end(),
            dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId]
              .begin(),
            dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId]
              .end(),
            std::inserter(s, s.begin()));
          mergedSet = s;
        }
      macroIdToNonlocalAtomsSetMap[cell] =
        std::vector<unsigned int>(mergedSet.begin(), mergedSet.end());
    }

  std::vector<unsigned int> nonlocalPseudoWfcsAccum(
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  std::vector<unsigned int> numPseudoWfcsAtom(
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  std::vector<std::vector<unsigned int>> projectorKetTimesVectorLocalIds(
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  unsigned int numPseudo = 0;
  for (unsigned int iAtom = 0;
       iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
       ++iAtom)
    {
      const unsigned int atomId =
        dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      nonlocalPseudoWfcsAccum[iAtom] = numPseudo;
      numPseudo += dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      numPseudoWfcsAtom[iAtom] =
        dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

      for (unsigned int ipsp = 0;
           ipsp < dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
           ++ipsp)
        projectorKetTimesVectorLocalIds[iAtom].push_back(
          dftPtr->d_projectorKetTimesVectorPar[0]
            .get_partitioner()
            ->global_to_local(
              dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(
                atomId, ipsp)]));
    }

  // band group parallelization data structures
  const unsigned int numberBandGroups =
    dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
  const unsigned int bandGroupTaskId =
    dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
  std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
  dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                             numEigenVectors,
                                             bandGroupLowHighPlusOneIndices);

  const unsigned int blockSize = std::min(dftParameters::chebyWfcBlockSize,
                                          bandGroupLowHighPlusOneIndices[1]);

  const unsigned int localVectorSize =
    dftPtr->d_eigenVectorsFlattenedSTL[0].size() / numEigenVectors;
  std::vector<std::vector<distributedCPUVec<double>>> eigenVectors(
    dftPtr->d_kPointWeights.size());
  std::vector<distributedCPUVec<dataTypes::number>> eigenVectorsFlattenedBlock(
    dftPtr->d_kPointWeights.size());

  const unsigned int numMacroCells    = matrixFreeData.n_macro_cells();
  const unsigned int numPhysicalCells = matrixFreeData.n_physical_cells();

#if defined(DFTFE_WITH_GPU)
  AssertThrow(
    numMacroCells == numPhysicalCells,
    ExcMessage(
      "DFT-FE Error: dealii for GPU DFT-FE must be compiled without any vectorization enabled."));

  // create map between macro cell id and normal cell id
  std::vector<unsigned int> normalCellIdToMacroCellIdMap(numPhysicalCells);
  std::vector<unsigned int> macroCellIdToNormalCellIdMap(numPhysicalCells);

  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
  unsigned int                                         iElemNormal = 0;
  for (const auto &cell :
       matrixFreeData.get_dof_handler().active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          bool         isFound        = false;
          unsigned int iElemMacroCell = 0;
          for (unsigned int iMacroCell = 0; iMacroCell < numMacroCells;
               ++iMacroCell)
            {
              const unsigned int n_sub_cells =
                matrixFreeData.n_components_filled(iMacroCell);
              for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
                {
                  cellPtr = matrixFreeData.get_cell_iterator(iMacroCell, iCell);
                  if (cell->id() == cellPtr->id())
                    {
                      normalCellIdToMacroCellIdMap[iElemNormal] =
                        iElemMacroCell;
                      macroCellIdToNormalCellIdMap[iElemMacroCell] =
                        iElemNormal;
                      isFound = true;
                      break;
                    }
                  iElemMacroCell++;
                }

              if (isFound)
                break;
            }
          iElemNormal++;
        }
    }

  std::vector<unsigned int> nonTrivialNonLocalIdsAllCells;
  std::vector<unsigned int> nonTrivialIdToElemIdMap;
  std::vector<unsigned int> nonTrivialIdToAllPseudoWfcIdMap;
  std::vector<unsigned int> projecterKetTimesFlattenedVectorLocalIds;
  if (isPseudopotential)
    {
      for (unsigned int ielem = 0; ielem < numPhysicalCells; ++ielem)
        {
          const unsigned int numNonLocalAtomsCurrentProc =
            dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

          const unsigned int macroCellId = normalCellIdToMacroCellIdMap[ielem];
          for (unsigned int iatom = 0; iatom < numNonLocalAtomsCurrentProc;
               ++iatom)
            {
              bool               isNonTrivial = false;
              const unsigned int macroCellId =
                normalCellIdToMacroCellIdMap[ielem];
              for (unsigned int i = 0;
                   i < macroIdToNonlocalAtomsSetMap[macroCellId].size();
                   i++)
                if (macroIdToNonlocalAtomsSetMap[macroCellId][i] == iatom)
                  {
                    isNonTrivial = true;
                    break;
                  }
              if (isNonTrivial)
                {
                  const int globalAtomId =
                    dftPtr->d_nonLocalAtomIdsInCurrentProcess[iatom];
                  const unsigned int numberSingleAtomPseudoWaveFunctions =
                    numPseudoWfcsAtom[iatom];
                  for (unsigned int ipsp = 0;
                       ipsp < numberSingleAtomPseudoWaveFunctions;
                       ++ipsp)
                    {
                      nonTrivialNonLocalIdsAllCells.push_back(iatom);
                      nonTrivialIdToElemIdMap.push_back(ielem);
                      nonTrivialIdToAllPseudoWfcIdMap.push_back(
                        nonlocalPseudoWfcsAccum[iatom] + ipsp);
                      // const unsigned int
                      // id=dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner()->global_to_local(dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(globalAtomId,ipsp)]);
                      projecterKetTimesFlattenedVectorLocalIds.push_back(
                        projectorKetTimesVectorLocalIds[iatom][ipsp]);
                    }
                }
            }
        }
    }
#endif

    // allocate storage for vector of quadPoints, nonlocal atom id, pseudo wave,
    // k point
    // FIXME: flatten nonlocal atomid id and pseudo wave and k point
#ifdef USE_COMPLEX
  std::vector<std::vector<std::vector<std::vector<
    std::vector<Tensor<1, 2, Tensor<1, C_DIM, VectorizedArray<double>>>>>>>>
    zetalmDeltaVlProductDistImageAtomsQuads;
#else
  std::vector<std::vector<
    std::vector<std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>>>>
    zetalmDeltaVlProductDistImageAtomsQuads;
#endif

  std::vector<std::vector<std::vector<dataTypes::number>>>
    projectorKetTimesPsiTimesVTimesPartOcc(numKPoints);

  if (isPseudopotential)
    {
      zetalmDeltaVlProductDistImageAtomsQuads.resize(
        matrixFreeData.n_macro_cells());
      for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
        {
          const unsigned int numSubCells =
            matrixFreeData.n_components_filled(cell);
          zetalmDeltaVlProductDistImageAtomsQuads[cell].resize(
            numQuadPointsNLP);
          for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
            {
              zetalmDeltaVlProductDistImageAtomsQuads[cell][q].resize(
                dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                  .size());
              for (unsigned int i = 0;
                   i <
                   dftPtr
                     ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                     .size();
                   ++i)
                {
                  const int numberPseudoWaveFunctions =
                    dftPtr
                      ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                        [i]
                      .size();
                  zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i].resize(
                    numberPseudoWaveFunctions);
                  for (unsigned int iPseudoWave = 0;
                       iPseudoWave < numberPseudoWaveFunctions;
                       ++iPseudoWave)
                    {
#ifdef USE_COMPLEX
                      zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i]
                                                             [iPseudoWave]
                                                               .resize(
                                                                 numKPoints,
                                                                 zeroTensor2);
#else
                      zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i]
                                                             [iPseudoWave] =
                                                               zeroTensor3;
#endif
                    }
                }
            }
        }

      for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
        {
          const unsigned int numSubCells =
            matrixFreeData.n_components_filled(cell);
          for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
            {
              subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
              dealii::CellId subCellId = subCellPtr->id();

              for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                {
                  for (
                    unsigned int i = 0;
                    i <
                    dftPtr
                      ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                      .size();
                    ++i)
                    {
                      const int numberPseudoWaveFunctions =
                        dftPtr
                          ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                            [i]
                          .size();
                      for (unsigned int iPseudoWave = 0;
                           iPseudoWave < numberPseudoWaveFunctions;
                           ++iPseudoWave)
                        {
                          if (
                            dftPtr
                              ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                                [i][iPseudoWave]
                              .find(subCellId) !=
                            dftPtr
                              ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                                [i][iPseudoWave]
                              .end())
                            {
                              for (unsigned int ikPoint = 0;
                                   ikPoint < numKPoints;
                                   ++ikPoint)
                                {
                                  for (unsigned int idim = 0; idim < C_DIM;
                                       idim++)
                                    {
#ifdef USE_COMPLEX
                                      zetalmDeltaVlProductDistImageAtomsQuads
                                        [cell][q][i][iPseudoWave][ikPoint][0]
                                        [idim][iSubCell] =
                                          dftPtr
                                            ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                                              [i][iPseudoWave][subCellId]
                                              [ikPoint * numQuadPointsNLP *
                                                 C_DIM * 2 +
                                               q * C_DIM * 2 + idim * 2 + 0];
                                      zetalmDeltaVlProductDistImageAtomsQuads
                                        [cell][q][i][iPseudoWave][ikPoint][1]
                                        [idim][iSubCell] =
                                          dftPtr
                                            ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                                              [i][iPseudoWave][subCellId]
                                              [ikPoint * numQuadPointsNLP *
                                                 C_DIM * 2 +
                                               q * C_DIM * 2 + idim * 2 + 1];
#else
                                      zetalmDeltaVlProductDistImageAtomsQuads
                                        [cell][q][i][iPseudoWave][idim]
                                        [iSubCell] =
                                          dftPtr
                                            ->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint
                                              [i][iPseudoWave][subCellId]
                                              [ikPoint * numQuadPointsNLP *
                                                 C_DIM +
                                               q * C_DIM + idim];
#endif
                                    }
                                }
                            } // non-trivial cellId check
                        }     // iPseudoWave loop
                    }         // i loop
                }             // q loop
            }                 // subcell loop
        }                     // macrocell loop
    }

  std::vector<std::vector<double>> partialOccupancies(
    dftPtr->d_kPointWeights.size(),
    std::vector<double>((1 + dftParameters::spinPolarized) * numEigenVectors,
                        0.0));
  for (unsigned int spinIndex = 0;
       spinIndex < (1 + dftParameters::spinPolarized);
       ++spinIndex)
    for (unsigned int kPoint = 0; kPoint < dftPtr->d_kPointWeights.size();
         ++kPoint)
      for (unsigned int iWave = 0; iWave < numEigenVectors; ++iWave)
        {
          const double eigenValue =
            dftPtr->eigenValues[kPoint][numEigenVectors * spinIndex + iWave];
          partialOccupancies[kPoint][numEigenVectors * spinIndex + iWave] =
            dftUtils::getPartialOccupancy(eigenValue,
                                          dftPtr->fermiEnergy,
                                          C_kb,
                                          dftParameters::TVal);

          if (dftParameters::constraintMagnetization)
            {
              partialOccupancies[kPoint][numEigenVectors * spinIndex + iWave] =
                1.0;
              if (spinIndex == 0)
                {
                  if (eigenValue > dftPtr->fermiEnergyUp)
                    partialOccupancies[kPoint][numEigenVectors * spinIndex +
                                               iWave] = 0.0;
                }
              else if (spinIndex == 1)
                {
                  if (eigenValue > dftPtr->fermiEnergyDown)
                    partialOccupancies[kPoint][numEigenVectors * spinIndex +
                                               iWave] = 0.0;
                }
            }
        }

  MPI_Barrier(MPI_COMM_WORLD);
  init_time = MPI_Wtime() - init_time;

  for (unsigned int spinIndex = 0;
       spinIndex < (1 + dftParameters::spinPolarized);
       ++spinIndex)
    {
#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
      std::vector<double>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened(
          nonTrivialNonLocalIdsAllCells.size() * numQuadPointsNLP * 3, 0.0);
      std::vector<double> elocWfcEshelbyTensorQuadValuesH(numPhysicalCells *
                                                            numQuadPoints * 6,
                                                          0.0);
#endif
      std::vector<std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads(
          numMacroCells * numQuadPointsNLP,
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>(numPseudo,
                                                                 zeroTensor3));

      if (dftParameters::useGPU)
        {
#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
          MPI_Barrier(MPI_COMM_WORLD);
          double gpu_time = MPI_Wtime();

          forceCUDA::gpuPortedForceKernelsAllH(
            kohnShamDFTEigenOperator,
            dftPtr->d_eigenVectorsFlattenedCUDA.begin() +
              spinIndex * localVectorSize * numEigenVectors,
            &dftPtr->eigenValues[0][spinIndex * numEigenVectors],
            &partialOccupancies[0][spinIndex * numEigenVectors],
            &nonTrivialIdToElemIdMap[0],
            &projecterKetTimesFlattenedVectorLocalIds[0],
            numEigenVectors,
            numPhysicalCells,
            numQuadPoints,
            numQuadPointsNLP,
            dftPtr->matrix_free_data.get_dofs_per_cell(
              dftPtr->d_densityDofHandlerIndex),
            nonTrivialNonLocalIdsAllCells.size(),
            &elocWfcEshelbyTensorQuadValuesH[0],
            &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
              [0],
            dftPtr->interBandGroupComm,
            isPseudopotential);

          MPI_Barrier(MPI_COMM_WORLD);
          gpu_time = MPI_Wtime() - gpu_time;

          if (this_process == 0 && dftParameters::verbosity >= 4)
            std::cout << "Time for gpuPortedForceKernelsAllH: " << gpu_time
                      << std::endl;
#endif
        }
      else
        {
          for (unsigned int ivec = 0; ivec < numEigenVectors; ivec += blockSize)
            {
              const unsigned int currentBlockSize =
                std::min(blockSize, numEigenVectors - ivec);

              if (currentBlockSize != blockSize || ivec == 0)
                {
                  for (unsigned int kPoint = 0;
                       kPoint < dftPtr->d_kPointWeights.size();
                       ++kPoint)
                    {
                      eigenVectors[kPoint].resize(currentBlockSize);
                      for (unsigned int i = 0; i < currentBlockSize; ++i)
                        eigenVectors[kPoint][i].reinit(dftPtr->d_tempEigenVec);


                      vectorTools::createDealiiVector<dataTypes::number>(
                        dftPtr->matrix_free_data.get_vector_partitioner(
                          dftPtr->d_densityDofHandlerIndex),
                        currentBlockSize,
                        eigenVectorsFlattenedBlock[kPoint]);
                      eigenVectorsFlattenedBlock[kPoint] =
                        dataTypes::number(0.0);
                    }

                  dftPtr->constraintsNoneDataInfo.precomputeMaps(
                    dftPtr->matrix_free_data.get_vector_partitioner(
                      dftPtr->d_densityDofHandlerIndex),
                    eigenVectorsFlattenedBlock[0].get_partitioner(),
                    currentBlockSize);
                }

              if ((ivec + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (ivec + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  std::vector<std::vector<double>> blockedEigenValues(
                    dftPtr->d_kPointWeights.size(),
                    std::vector<double>(currentBlockSize, 0.0));
                  std::vector<std::vector<double>> blockedPartialOccupancies(
                    dftPtr->d_kPointWeights.size(),
                    std::vector<double>(currentBlockSize, 0.0));
                  for (unsigned int kPoint = 0;
                       kPoint < dftPtr->d_kPointWeights.size();
                       ++kPoint)
                    for (unsigned int iWave = 0; iWave < currentBlockSize;
                         ++iWave)
                      {
                        blockedEigenValues[kPoint][iWave] =
                          dftPtr
                            ->eigenValues[kPoint][numEigenVectors * spinIndex +
                                                  ivec + iWave];
                        blockedPartialOccupancies[kPoint][iWave] =
                          partialOccupancies[kPoint]
                                            [numEigenVectors * spinIndex +
                                             ivec + iWave];
                      }

                  for (unsigned int kPoint = 0;
                       kPoint < dftPtr->d_kPointWeights.size();
                       ++kPoint)
                    {
                      for (unsigned int iNode = 0; iNode < localVectorSize;
                           ++iNode)
                        for (unsigned int iWave = 0; iWave < currentBlockSize;
                             ++iWave)
                          eigenVectorsFlattenedBlock[kPoint].local_element(
                            iNode * currentBlockSize + iWave) =
                            dftPtr->d_eigenVectorsFlattenedSTL
                              [(dftParameters::spinPolarized + 1) * kPoint +
                               spinIndex]
                              [iNode * numEigenVectors + ivec + iWave];

                      dftPtr->constraintsNoneDataInfo.distribute(
                        eigenVectorsFlattenedBlock[kPoint], currentBlockSize);
                      eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
                      vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                        eigenVectorsFlattenedBlock[kPoint],
                        currentBlockSize,
                        std::make_pair(0, currentBlockSize),
                        dftPtr->localProc_dof_indicesReal,
                        dftPtr->localProc_dof_indicesImag,
                        eigenVectors[kPoint],
                        false);

                      // FIXME: The underlying call to update_ghost_values
                      // is required because currently localProc_dof_indicesReal
                      // and localProc_dof_indicesImag are only available for
                      // locally owned nodes. Once they are also made available
                      // for ghost nodes- use true for the last argument in
                      // copyFlattenedDealiiVecToSingleCompVec(..) above and
                      // supress underlying call.
                      for (unsigned int i = 0; i < currentBlockSize; ++i)
                        eigenVectors[kPoint][i].update_ghost_values();
#else
                      vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                        eigenVectorsFlattenedBlock[kPoint],
                        currentBlockSize,
                        std::make_pair(0, currentBlockSize),
                        eigenVectors[kPoint],
                        true);

#endif
                    }

                  if (isPseudopotential)
                    for (unsigned int ikPoint = 0; ikPoint < numKPoints;
                         ++ikPoint)
                      {
                        computeNonLocalProjectorKetTimesPsiTimesVFlattened(
                          eigenVectorsFlattenedBlock[ikPoint],
                          currentBlockSize,
                          projectorKetTimesPsiTimesVTimesPartOcc[ikPoint],
                          ikPoint,
                          blockedPartialOccupancies[ikPoint]
#ifdef USE_COMPLEX
                          ,
                          true
#endif
                        );
                      }

                  for (unsigned int cell = 0;
                       cell < matrixFreeData.n_macro_cells();
                       ++cell)
                    {
                      forceEval.reinit(cell);
#ifdef USE_COMPLEX
                      forceEvalKPoints.reinit(cell);
#endif

                      psiEval.reinit(cell);

                      forceEvalNLP.reinit(cell);
#ifdef USE_COMPLEX
                      forceEvalKPointsNLP.reinit(cell);
#endif

                      psiEvalNLP.reinit(cell);

                      const unsigned int numSubCells =
                        matrixFreeData.n_components_filled(cell);
#ifdef USE_COMPLEX
                      std::vector<Tensor<1, 2, VectorizedArray<double>>>
                        psiQuads(numQuadPoints * currentBlockSize * numKPoints,
                                 zeroTensor1);
                      std::vector<
                        Tensor<1, 2, Tensor<1, C_DIM, VectorizedArray<double>>>>
                        gradPsiQuads(numQuadPoints * currentBlockSize *
                                       numKPoints,
                                     zeroTensor2);
#else
                      std::vector<VectorizedArray<double>> psiQuads(
                        numQuadPoints * currentBlockSize,
                        make_vectorized_array(0.0));
                      std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
                                                           gradPsiQuads(numQuadPoints * currentBlockSize,
                                     zeroTensor3);
#endif

                      for (unsigned int ikPoint = 0; ikPoint < numKPoints;
                           ++ikPoint)
                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            psiEval.read_dof_values_plain(
                              eigenVectors[ikPoint][iEigenVec]);
                            psiEval.evaluate(true, true);

                            for (unsigned int q = 0; q < numQuadPoints; ++q)
                              {
                                const unsigned int id =
                                  q * currentBlockSize * numKPoints +
                                  currentBlockSize * ikPoint + iEigenVec;
                                psiQuads[id]     = psiEval.get_value(q);
                                gradPsiQuads[id] = psiEval.get_gradient(q);
                              } // quad point loop
                          }     // eigenvector loop

#ifdef USE_COMPLEX
                      std::vector<Tensor<1, 2, VectorizedArray<double>>>
                        psiQuadsNLP;
                      std::vector<
                        Tensor<1, 2, Tensor<1, C_DIM, VectorizedArray<double>>>>
                        gradPsiQuadsNLP;
#else
                      std::vector<VectorizedArray<double>> psiQuadsNLP;
                      std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
                        gradPsiQuadsNLP;
#endif

                      if (isPseudopotential)
                        {
#ifdef USE_COMPLEX
                          psiQuadsNLP.resize(numQuadPointsNLP *
                                               currentBlockSize * numKPoints,
                                             zeroTensor1);
                          gradPsiQuadsNLP.resize(numQuadPointsNLP *
                                                   currentBlockSize *
                                                   numKPoints,
                                                 zeroTensor2);
#else
                          psiQuadsNLP.resize(numQuadPointsNLP *
                                               currentBlockSize,
                                             make_vectorized_array(0.0));
                          gradPsiQuadsNLP.resize(numQuadPointsNLP *
                                                   currentBlockSize *
                                                   numKPoints,
                                                 zeroTensor3);
#endif

                          for (unsigned int ikPoint = 0; ikPoint < numKPoints;
                               ++ikPoint)
                            for (unsigned int iEigenVec = 0;
                                 iEigenVec < currentBlockSize;
                                 ++iEigenVec)
                              {
                                psiEvalNLP.read_dof_values_plain(
                                  eigenVectors[ikPoint][iEigenVec]);
                                psiEvalNLP.evaluate(true, true);

                                for (unsigned int q = 0; q < numQuadPointsNLP;
                                     ++q)
                                  {
                                    const unsigned int id =
                                      q * currentBlockSize * numKPoints +
                                      currentBlockSize * ikPoint + iEigenVec;
                                    psiQuadsNLP[id] = psiEvalNLP.get_value(q);
                                    gradPsiQuadsNLP[id] =
                                      psiEvalNLP.get_gradient(q);
                                  } // quad point loop
                              }     // eigenvector loop
                        }

#ifndef USE_COMPLEX
                      const unsigned int numNonLocalAtomsCurrentProc =
                        projectorKetTimesPsiTimesVTimesPartOcc[0].size();
                      std::vector<bool> isAtomInCell(
                        numNonLocalAtomsCurrentProc, false);
                      if (isPseudopotential)
                        {
                          std::vector<unsigned int> nonTrivialNonLocalIds;
                          for (unsigned int iatom = 0;
                               iatom < numNonLocalAtomsCurrentProc;
                               ++iatom)
                            {
                              for (unsigned int i = 0;
                                   i <
                                   macroIdToNonlocalAtomsSetMap[cell].size();
                                   i++)
                                if (macroIdToNonlocalAtomsSetMap[cell][i] ==
                                    iatom)
                                  {
                                    isAtomInCell[iatom] = true;
                                    nonTrivialNonLocalIds.push_back(iatom);
                                    break;
                                  }
                            }

                          for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                            {
                              std::vector<
                                Tensor<1,
                                       C_DIM,
                                       VectorizedArray<double>>> &tempContract =
                                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                                  [cell * numQuadPointsNLP + q];
                              // std::fill(temp1.begin(),temp1.end(),make_vectorized_array(0.0));
                              for (unsigned int i = 0;
                                   i < nonTrivialNonLocalIds.size();
                                   ++i)
                                {
                                  const unsigned int iatom =
                                    nonTrivialNonLocalIds[i];
                                  const unsigned int
                                    numberSingleAtomPseudoWaveFunctions =
                                      numPseudoWfcsAtom[iatom];
                                  const unsigned int startingId =
                                    nonlocalPseudoWfcsAccum[iatom];
                                  const std::vector<double> &temp2 =
                                    projectorKetTimesPsiTimesVTimesPartOcc
                                      [0][iatom];
                                  for (unsigned int ipsp = 0;
                                       ipsp <
                                       numberSingleAtomPseudoWaveFunctions;
                                       ++ipsp)
                                    for (unsigned int iEigenVec = 0;
                                         iEigenVec < currentBlockSize;
                                         ++iEigenVec)
                                      {
                                        tempContract[startingId + ipsp] +=
                                          gradPsiQuadsNLP[q * currentBlockSize +
                                                          iEigenVec] *
                                          make_vectorized_array(
                                            temp2[ipsp * currentBlockSize +
                                                  iEigenVec]);
                                      }
                                }
                            }
                        }

#endif
                      Tensor<2, C_DIM, VectorizedArray<double>>
                        EKPointsQuadSum = zeroTensor4;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
#ifdef USE_COMPLEX
                          Tensor<2, C_DIM, VectorizedArray<double>> EKPoints =
                            eshelbyTensor::
                              getELocWfcEshelbyTensorPeriodicKPoints(
                                psiQuads.begin() +
                                  q * currentBlockSize * numKPoints,
                                gradPsiQuads.begin() +
                                  q * currentBlockSize * numKPoints,
                                dftPtr->d_kPointCoordinates,
                                dftPtr->d_kPointWeights,
                                blockedEigenValues,
                                dftPtr->fermiEnergy,
                                dftParameters::TVal);

                          EKPoints += eshelbyTensor::getEKStress(
                            psiQuads.begin() +
                              q * currentBlockSize * numKPoints,
                            gradPsiQuads.begin() +
                              q * currentBlockSize * numKPoints,
                            dftPtr->d_kPointCoordinates,
                            dftPtr->d_kPointWeights,
                            blockedEigenValues,
                            dftPtr->fermiEnergy,
                            dftParameters::TVal);
#else
                          Tensor<2, C_DIM, VectorizedArray<double>> EKPoints =
                            eshelbyTensor::getELocWfcEshelbyTensorNonPeriodic(
                              psiQuads.begin() +
                                q * currentBlockSize * numKPoints,
                              gradPsiQuads.begin() +
                                q * currentBlockSize * numKPoints,
                              blockedEigenValues[0],
                              blockedPartialOccupancies[0]);
#endif

                          EKPointsQuadSum += EKPoints * forceEval.JxW(q);
                        } // quad point loop

                      if (isPseudopotential)
                        {
                          for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                            {
#ifdef USE_COMPLEX
                              Tensor<2, C_DIM, VectorizedArray<double>> Enl =
                                eshelbyTensor::getEnlStress(
                                  zetalmDeltaVlProductDistImageAtomsQuads[cell]
                                                                         [q],
                                  projectorKetTimesPsiTimesVTimesPartOcc,
                                  psiQuadsNLP.begin() +
                                    q * currentBlockSize * numKPoints,
                                  gradPsiQuadsNLP.begin() +
                                    q * currentBlockSize * numKPoints,
                                  dftPtr->d_kPointWeights,
                                  dftPtr->d_kPointCoordinates,
                                  macroIdToNonlocalAtomsSetMap[cell],
                                  currentBlockSize);
#else

                              Tensor<2, C_DIM, VectorizedArray<double>> Enl =
                                zeroTensor4;
                              /*
                                =eshelbyTensor::getEnlStress(zetalmDeltaVlProductDistImageAtomsQuads[cell][q],
                                  projectorKetTimesPsiTimesVTimesPartOcc,
                                  psiQuadsNLP.begin()+q*currentBlockSize*numKPoints,
                                  gradPsiQuadsNLP.begin()+q*currentBlockSize*numKPoints,
                                  macroIdToNonlocalAtomsSetMap[cell],
                                  currentBlockSize);
                                */
#endif


                              EKPointsQuadSum += Enl * forceEvalNLP.JxW(q);
                            }
                        }

                      for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                           ++iSubCell)
                        for (unsigned int idim = 0; idim < C_DIM; ++idim)
                          for (unsigned int jdim = 0; jdim < C_DIM; ++jdim)
                            {
                              d_stressKPoints[idim][jdim] +=
                                spinPolarizedFactor *
                                EKPointsQuadSum[idim][jdim][iSubCell];
                            }

                    } // macro cell loop
                }     // band parallelization loop
            }         // wavefunction block loop
        }

#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
      if (dftParameters::useGPU)
        {
          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              forceEval.reinit(cell);

              Tensor<2, C_DIM, VectorizedArray<double>> EKPointsQuadSum =
                zeroTensor4;
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  Tensor<2, C_DIM, VectorizedArray<double>> E;
                  const unsigned int                        physicalCellId =
                    macroCellIdToNormalCellIdMap[cell];
                  const unsigned int id = physicalCellId * numQuadPoints + q;
                  E[0][0]               = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 0]);
                  E[1][0] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 1]);
                  E[1][1] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 2]);
                  E[2][0] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 3]);
                  E[2][1] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 4]);
                  E[2][2] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 5]);
                  E[0][1] = E[1][0];
                  E[0][2] = E[2][0];
                  E[1][2] = E[2][1];

                  EKPointsQuadSum += E * forceEval.JxW(q);
                } // quad point loop

              const unsigned int numSubCells =
                matrixFreeData.n_components_filled(cell);
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                for (unsigned int idim = 0; idim < C_DIM; ++idim)
                  for (unsigned int jdim = 0; jdim < C_DIM; ++jdim)
                    {
                      d_stressKPoints[idim][jdim] +=
                        spinPolarizedFactor *
                        EKPointsQuadSum[idim][jdim][iSubCell];
                    }
            }

          if (isPseudopotential)
            for (unsigned int i = 0; i < nonTrivialNonLocalIdsAllCells.size();
                 ++i)
              {
                const unsigned int cell =
                  normalCellIdToMacroCellIdMap[nonTrivialIdToElemIdMap[i]];
                const unsigned int id = nonTrivialIdToAllPseudoWfcIdMap[i];
                for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                  {
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q][id]
                      [0] = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                          [i * numQuadPointsNLP * 3 + 3 * q + 0]);
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q][id]
                      [1] = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                          [i * numQuadPointsNLP * 3 + 3 * q + 1]);
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q][id]
                      [2] = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                          [i * numQuadPointsNLP * 3 + 3 * q + 2]);
                  }
              }
        }
#endif

#ifndef USE_COMPLEX
      if (isPseudopotential)
        for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
             ++cell)
          {
            forceEvalNLP.reinit(cell);

            const unsigned int numNonLocalAtomsCurrentProc =
              dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
            std::vector<bool> isAtomInCell(numNonLocalAtomsCurrentProc, false);

            std::vector<unsigned int> nonTrivialNonLocalIds;
            for (unsigned int iatom = 0; iatom < numNonLocalAtomsCurrentProc;
                 ++iatom)
              {
                for (unsigned int i = 0;
                     i < macroIdToNonlocalAtomsSetMap[cell].size();
                     i++)
                  if (macroIdToNonlocalAtomsSetMap[cell][i] == iatom)
                    {
                      isAtomInCell[iatom] = true;
                      nonTrivialNonLocalIds.push_back(iatom);
                      break;
                    }
              }

            Tensor<2, C_DIM, VectorizedArray<double>> EKPointsQuadSum =
              zeroTensor4;
            for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
              {
                Tensor<2, C_DIM, VectorizedArray<double>> Enl =
                  eshelbyTensor::getEnlStress(
                    zetalmDeltaVlProductDistImageAtomsQuads[cell][q],
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q],
                    isAtomInCell,
                    nonlocalPseudoWfcsAccum);
                EKPointsQuadSum += Enl * forceEvalNLP.JxW(q);
              }

            const unsigned int numSubCells =
              matrixFreeData.n_components_filled(cell);
            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              for (unsigned int idim = 0; idim < C_DIM; ++idim)
                for (unsigned int jdim = 0; jdim < C_DIM; ++jdim)
                  {
                    d_stressKPoints[idim][jdim] +=
                      spinPolarizedFactor *
                      EKPointsQuadSum[idim][jdim][iSubCell];
                  }
          }
#endif

    } // spin index

  MPI_Barrier(MPI_COMM_WORLD);
  double enowfc_time = MPI_Wtime();

  /////////// Compute contribution independent of wavefunctions
  ////////////////////
  if (bandGroupTaskId == 0)
    {
      if (dftParameters::spinPolarized == 1)
        {
          std::vector<VectorizedArray<double>> rhoXCQuadsVect(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
            gradRhoSpin0QuadsVect(numQuadPoints, zeroTensor3);
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
                                               gradRhoSpin1QuadsVect(numQuadPoints, zeroTensor3);
          std::vector<VectorizedArray<double>> excQuads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<VectorizedArray<double>> vxcRhoOutSpin0Quads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<VectorizedArray<double>> vxcRhoOutSpin1Quads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
            derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints,
                                                      zeroTensor3);
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
            derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints,
                                                      zeroTensor3);
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
            gradRhoCoreQuads(numQuadPoints, zeroTensor3);
          std::vector<Tensor<2, C_DIM, VectorizedArray<double>>>
            hessianRhoCoreQuads(numQuadPoints, zeroTensor4);

          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              forceEval.reinit(cell);

              std::fill(rhoXCQuadsVect.begin(),
                        rhoXCQuadsVect.end(),
                        make_vectorized_array(0.0));
              std::fill(gradRhoSpin0QuadsVect.begin(),
                        gradRhoSpin0QuadsVect.end(),
                        zeroTensor3);
              std::fill(gradRhoSpin1QuadsVect.begin(),
                        gradRhoSpin1QuadsVect.end(),
                        zeroTensor3);
              std::fill(excQuads.begin(),
                        excQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(vxcRhoOutSpin0Quads.begin(),
                        vxcRhoOutSpin0Quads.end(),
                        make_vectorized_array(0.0));
              std::fill(vxcRhoOutSpin1Quads.begin(),
                        vxcRhoOutSpin1Quads.end(),
                        make_vectorized_array(0.0));
              std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),
                        derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),
                        zeroTensor3);
              std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),
                        derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),
                        zeroTensor3);
              std::fill(gradRhoCoreQuads.begin(),
                        gradRhoCoreQuads.end(),
                        zeroTensor3);
              std::fill(hessianRhoCoreQuads.begin(),
                        hessianRhoCoreQuads.end(),
                        zeroTensor4);

              const unsigned int numSubCells =
                matrixFreeData.n_components_filled(cell);
              // For LDA
              std::vector<double> exchValRhoOut(numQuadPoints);
              std::vector<double> corrValRhoOut(numQuadPoints);
              std::vector<double> exchPotValRhoOut(2 * numQuadPoints);
              std::vector<double> corrPotValRhoOut(2 * numQuadPoints);
              std::vector<double> rhoOutQuadsXC(2 * numQuadPoints);

              //
              // For GGA
              std::vector<double> sigmaValRhoOut(3 * numQuadPoints);
              std::vector<double> derExchEnergyWithDensityValRhoOut(
                2 * numQuadPoints),
                derCorrEnergyWithDensityValRhoOut(2 * numQuadPoints),
                derExchEnergyWithSigmaRhoOut(3 * numQuadPoints),
                derCorrEnergyWithSigmaRhoOut(3 * numQuadPoints);
              std::vector<Tensor<1, C_DIM, double>> gradRhoOutQuadsXCSpin0(
                numQuadPoints);
              std::vector<Tensor<1, C_DIM, double>> gradRhoOutQuadsXCSpin1(
                numQuadPoints);

              //
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
                  dealii::CellId subCellId = subCellPtr->id();

                  const std::vector<double> &temp =
                    (*dftPtr->rhoOutValues).find(subCellId)->second;
                  const std::vector<double> &temp1 =
                    (*dftPtr->rhoOutValuesSpinPolarized)
                      .find(subCellId)
                      ->second;

                  rhoOutQuadsXC = temp1;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      rhoXCQuadsVect[q][iSubCell] = temp[q];
                    }

                  if (dftParameters::nonLinearCoreCorrection)
                    {
                      const std::vector<double> &temp2 =
                        rhoCoreValues.find(subCellId)->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          rhoOutQuadsXC[2 * q + 0] += temp2[q] / 2.0;
                          rhoOutQuadsXC[2 * q + 1] += temp2[q] / 2.0;
                          rhoXCQuadsVect[q][iSubCell] += temp2[q];
                        }
                    }

                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      const std::vector<double> &temp3 =
                        (*dftPtr->gradRhoOutValuesSpinPolarized)
                          .find(subCellId)
                          ->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        for (unsigned int idim = 0; idim < C_DIM; idim++)
                          {
                            gradRhoOutQuadsXCSpin0[q][idim] =
                              temp3[6 * q + idim];
                            gradRhoOutQuadsXCSpin1[q][idim] =
                              temp3[6 * q + 3 + idim];
                            gradRhoSpin0QuadsVect[q][idim][iSubCell] =
                              temp3[6 * q + idim];
                            gradRhoSpin1QuadsVect[q][idim][iSubCell] =
                              temp3[6 * q + 3 + idim];
                          }

                      if (dftParameters::nonLinearCoreCorrection)
                        {
                          const std::vector<double> &temp4 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < C_DIM; idim++)
                              {
                                gradRhoOutQuadsXCSpin0[q][idim] +=
                                  temp4[3 * q + idim] / 2.0;
                                gradRhoOutQuadsXCSpin1[q][idim] +=
                                  temp4[3 * q + idim] / 2.0;
                              }
                        }
                    }

                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          sigmaValRhoOut[3 * q + 0] =
                            scalar_product(gradRhoOutQuadsXCSpin0[q],
                                           gradRhoOutQuadsXCSpin0[q]);
                          sigmaValRhoOut[3 * q + 1] =
                            scalar_product(gradRhoOutQuadsXCSpin0[q],
                                           gradRhoOutQuadsXCSpin1[q]);
                          sigmaValRhoOut[3 * q + 2] =
                            scalar_product(gradRhoOutQuadsXCSpin1[q],
                                           gradRhoOutQuadsXCSpin1[q]);
                        }

                      xc_gga_exc_vxc(&(dftPtr->funcX),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &exchValRhoOut[0],
                                     &derExchEnergyWithDensityValRhoOut[0],
                                     &derExchEnergyWithSigmaRhoOut[0]);
                      xc_gga_exc_vxc(&(dftPtr->funcC),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &corrValRhoOut[0],
                                     &derCorrEnergyWithDensityValRhoOut[0],
                                     &derCorrEnergyWithSigmaRhoOut[0]);

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          excQuads[q][iSubCell] =
                            exchValRhoOut[q] + corrValRhoOut[q];
                          vxcRhoOutSpin0Quads[q][iSubCell] =
                            derExchEnergyWithDensityValRhoOut[2 * q] +
                            derCorrEnergyWithDensityValRhoOut[2 * q];
                          vxcRhoOutSpin1Quads[q][iSubCell] =
                            derExchEnergyWithDensityValRhoOut[2 * q + 1] +
                            derCorrEnergyWithDensityValRhoOut[2 * q + 1];
                          for (unsigned int idim = 0; idim < C_DIM; idim++)
                            {
                              derExchCorrEnergyWithGradRhoOutSpin0Quads
                                [q][idim][iSubCell] =
                                  2.0 *
                                  (derExchEnergyWithSigmaRhoOut[3 * q + 0] +
                                   derCorrEnergyWithSigmaRhoOut[3 * q + 0]) *
                                  gradRhoOutQuadsXCSpin0[q][idim];
                              derExchCorrEnergyWithGradRhoOutSpin0Quads
                                [q][idim][iSubCell] +=
                                (derExchEnergyWithSigmaRhoOut[3 * q + 1] +
                                 derCorrEnergyWithSigmaRhoOut[3 * q + 1]) *
                                gradRhoOutQuadsXCSpin1[q][idim];

                              derExchCorrEnergyWithGradRhoOutSpin1Quads
                                [q][idim][iSubCell] +=
                                2.0 *
                                (derExchEnergyWithSigmaRhoOut[3 * q + 2] +
                                 derCorrEnergyWithSigmaRhoOut[3 * q + 2]) *
                                gradRhoOutQuadsXCSpin1[q][idim];
                              derExchCorrEnergyWithGradRhoOutSpin1Quads
                                [q][idim][iSubCell] +=
                                (derExchEnergyWithSigmaRhoOut[3 * q + 1] +
                                 derCorrEnergyWithSigmaRhoOut[3 * q + 1]) *
                                gradRhoOutQuadsXCSpin0[q][idim];
                            }
                        }
                    }
                  else
                    {
                      xc_lda_exc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchValRhoOut[0]);
                      xc_lda_exc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchPotValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrPotValRhoOut[0]);
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          excQuads[q][iSubCell] =
                            exchValRhoOut[q] + corrValRhoOut[q];
                          vxcRhoOutSpin0Quads[q][iSubCell] =
                            exchPotValRhoOut[2 * q] + corrPotValRhoOut[2 * q];
                          vxcRhoOutSpin1Quads[q][iSubCell] =
                            exchPotValRhoOut[2 * q + 1] +
                            corrPotValRhoOut[2 * q + 1];
                        }
                    }

                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      if (dftParameters::nonLinearCoreCorrection == true)
                        {
                          const std::vector<double> &temp1 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < C_DIM; idim++)
                              gradRhoCoreQuads[q][idim][iSubCell] =
                                temp1[3 * q + idim] / 2.0;

                          if (dftParameters::xcFamilyType == "GGA")
                            {
                              const std::vector<double> &temp2 =
                                hessianRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < C_DIM;
                                     ++idim)
                                  for (unsigned int jdim = 0; jdim < C_DIM;
                                       ++jdim)
                                    hessianRhoCoreQuads
                                      [q][idim][jdim][iSubCell] =
                                        temp2[9 * q + 3 * idim + jdim] / 2.0;
                            }
                        }
                    }

                } // subcell loop

              Tensor<2, C_DIM, VectorizedArray<double>> EQuadSum = zeroTensor4;
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  Tensor<2, C_DIM, VectorizedArray<double>> E =
                    eshelbyTensorSP::getELocXcEshelbyTensor(
                      rhoXCQuadsVect[q],
                      gradRhoSpin0QuadsVect[q],
                      gradRhoSpin1QuadsVect[q],
                      excQuads[q],
                      derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                      derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

                  EQuadSum += E * forceEval.JxW(q);
                } // quad point loop

              if (isPseudopotential)
                {
                  if (dftParameters::nonLinearCoreCorrection)
                    addENonlinearCoreCorrectionStressContributionSpinPolarized(
                      forceEval,
                      matrixFreeData,
                      cell,
                      vxcRhoOutSpin0Quads,
                      vxcRhoOutSpin1Quads,
                      derExchCorrEnergyWithGradRhoOutSpin0Quads,
                      derExchCorrEnergyWithGradRhoOutSpin1Quads,
                      gradRhoCoreAtoms,
                      hessianRhoCoreAtoms,
                      dftParameters::xcFamilyType == "GGA");
                }

              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                for (unsigned int idim = 0; idim < C_DIM; ++idim)
                  for (unsigned int jdim = 0; jdim < C_DIM; ++jdim)
                    {
                      d_stress[idim][jdim] += EQuadSum[idim][jdim][iSubCell];
                    }
            } // macrocell loop
        }
      else
        {
          std::vector<VectorizedArray<double>> rhoQuads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<VectorizedArray<double>> rhoXCQuads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<VectorizedArray<double>> phiTotRhoOutQuads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>> gradRhoQuads(
            numQuadPoints, zeroTensor3);
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
            gradRhoCoreQuads(numQuadPoints, zeroTensor3);
          std::vector<Tensor<2, C_DIM, VectorizedArray<double>>>
                                               hessianRhoCoreQuads(numQuadPoints, zeroTensor4);
          std::vector<VectorizedArray<double>> excQuads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<VectorizedArray<double>> vxcRhoOutQuads(
            numQuadPoints, make_vectorized_array(0.0));
          std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
            derExchCorrEnergyWithGradRhoOutQuads(numQuadPoints, zeroTensor3);

          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              forceEval.reinit(cell);

              std::fill(rhoQuads.begin(),
                        rhoQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(rhoXCQuads.begin(),
                        rhoXCQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(phiTotRhoOutQuads.begin(),
                        phiTotRhoOutQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(gradRhoQuads.begin(), gradRhoQuads.end(), zeroTensor3);
              std::fill(gradRhoCoreQuads.begin(),
                        gradRhoCoreQuads.end(),
                        zeroTensor3);
              std::fill(hessianRhoCoreQuads.begin(),
                        hessianRhoCoreQuads.end(),
                        zeroTensor4);
              std::fill(excQuads.begin(),
                        excQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(vxcRhoOutQuads.begin(),
                        vxcRhoOutQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(derExchCorrEnergyWithGradRhoOutQuads.begin(),
                        derExchCorrEnergyWithGradRhoOutQuads.end(),
                        zeroTensor3);

              const unsigned int numSubCells =
                matrixFreeData.n_components_filled(cell);
              // For LDA
              std::vector<double> exchValRhoOut(numQuadPoints);
              std::vector<double> corrValRhoOut(numQuadPoints);
              std::vector<double> exchPotValRhoOut(numQuadPoints);
              std::vector<double> corrPotValRhoOut(numQuadPoints);
              std::vector<double> rhoOutQuadsXC(numQuadPoints);

              //
              // For GGA
              std::vector<double> sigmaValRhoOut(numQuadPoints);
              std::vector<double> derExchEnergyWithDensityValRhoOut(
                numQuadPoints),
                derCorrEnergyWithDensityValRhoOut(numQuadPoints),
                derExchEnergyWithSigmaRhoOut(numQuadPoints),
                derCorrEnergyWithSigmaRhoOut(numQuadPoints);
              std::vector<Tensor<1, C_DIM, double>> gradRhoOutQuadsXC(
                numQuadPoints);

              //
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
                  dealii::CellId subCellId = subCellPtr->id();

                  const std::vector<double> &temp1 =
                    rhoOutValues.find(subCellId)->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      rhoOutQuadsXC[q]        = temp1[q];
                      rhoQuads[q][iSubCell]   = temp1[q];
                      rhoXCQuads[q][iSubCell] = temp1[q];
                    }

                  if (dftParameters::nonLinearCoreCorrection)
                    {
                      const std::vector<double> &temp2 =
                        rhoCoreValues.find(subCellId)->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          rhoOutQuadsXC[q] += temp2[q];
                          rhoXCQuads[q][iSubCell] += temp2[q];
                        }
                    }

                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      const std::vector<double> &temp3 =
                        gradRhoOutValues.find(subCellId)->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        for (unsigned int idim = 0; idim < C_DIM; idim++)
                          {
                            gradRhoOutQuadsXC[q][idim] = temp3[3 * q + idim];
                            gradRhoQuads[q][idim][iSubCell] =
                              temp3[3 * q + idim];
                          }

                      if (dftParameters::nonLinearCoreCorrection)
                        {
                          const std::vector<double> &temp4 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              gradRhoOutQuadsXC[q][0] += temp4[3 * q + 0];
                              gradRhoOutQuadsXC[q][1] += temp4[3 * q + 1];
                              gradRhoOutQuadsXC[q][2] += temp4[3 * q + 2];
                            }
                        }
                    }

                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        sigmaValRhoOut[q] = gradRhoOutQuadsXC[q].norm_square();

                      xc_gga_exc_vxc(&(dftPtr->funcX),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &exchValRhoOut[0],
                                     &derExchEnergyWithDensityValRhoOut[0],
                                     &derExchEnergyWithSigmaRhoOut[0]);
                      xc_gga_exc_vxc(&(dftPtr->funcC),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &corrValRhoOut[0],
                                     &derCorrEnergyWithDensityValRhoOut[0],
                                     &derCorrEnergyWithSigmaRhoOut[0]);

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          excQuads[q][iSubCell] =
                            exchValRhoOut[q] + corrValRhoOut[q];
                          vxcRhoOutQuads[q][iSubCell] =
                            derExchEnergyWithDensityValRhoOut[q] +
                            derCorrEnergyWithDensityValRhoOut[q];

                          for (unsigned int idim = 0; idim < C_DIM; idim++)
                            {
                              derExchCorrEnergyWithGradRhoOutQuads
                                [q][idim][iSubCell] =
                                  2.0 *
                                  (derExchEnergyWithSigmaRhoOut[q] +
                                   derCorrEnergyWithSigmaRhoOut[q]) *
                                  gradRhoOutQuadsXC[q][idim];
                            }
                        }
                    }
                  else
                    {
                      xc_lda_exc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchValRhoOut[0]);
                      xc_lda_exc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchPotValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrPotValRhoOut[0]);

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          excQuads[q][iSubCell] =
                            exchValRhoOut[q] + corrValRhoOut[q];
                          vxcRhoOutQuads[q][iSubCell] =
                            exchPotValRhoOut[q] + corrPotValRhoOut[q];
                        }
                    }

                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      if (dftParameters::nonLinearCoreCorrection == true)
                        {
                          const std::vector<double> &temp1 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < C_DIM; idim++)
                              gradRhoCoreQuads[q][idim][iSubCell] =
                                temp1[3 * q + idim];

                          if (dftParameters::xcFamilyType == "GGA")
                            {
                              const std::vector<double> &temp2 =
                                hessianRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < C_DIM;
                                     ++idim)
                                  for (unsigned int jdim = 0; jdim < C_DIM;
                                       ++jdim)
                                    hessianRhoCoreQuads[q][idim][jdim]
                                                       [iSubCell] =
                                                         temp2[9 * q +
                                                               3 * idim + jdim];
                            }
                        }
                    }
                } // subcell loop

              Tensor<2, C_DIM, VectorizedArray<double>> EQuadSum = zeroTensor4;
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  Tensor<2, C_DIM, VectorizedArray<double>> E =
                    eshelbyTensor::getELocXcEshelbyTensor(
                      rhoXCQuads[q],
                      gradRhoQuads[q],
                      excQuads[q],
                      derExchCorrEnergyWithGradRhoOutQuads[q]);

                  EQuadSum += E * forceEval.JxW(q);
                } // quad point loop

              if (isPseudopotential)
                {
                  if (dftParameters::nonLinearCoreCorrection)
                    addENonlinearCoreCorrectionStressContribution(
                      forceEval,
                      matrixFreeData,
                      cell,
                      vxcRhoOutQuads,
                      derExchCorrEnergyWithGradRhoOutQuads,
                      gradRhoCoreAtoms,
                      hessianRhoCoreAtoms);
                }

              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                for (unsigned int idim = 0; idim < C_DIM; ++idim)
                  for (unsigned int jdim = 0; jdim < C_DIM; ++jdim)
                    {
                      d_stress[idim][jdim] += EQuadSum[idim][jdim][iSubCell];
                    }
            } // cell loop
        }

      ////Add electrostatic stress contribution////////////////
      computeStressEEshelbyEElectroPhiTot(matrixFreeDataElectro,
                                          phiTotDofHandlerIndexElectro,
                                          smearedChargeQuadratureId,
                                          lpspQuadratureIdElectro,
                                          phiTotRhoOutElectro,
                                          rhoOutValuesElectro,
                                          rhoOutValuesElectroLpsp,
                                          gradRhoOutValuesElectro,
                                          gradRhoOutValuesElectroLpsp,
                                          pseudoVLocElectro,
                                          pseudoVLocAtomsElectro,
                                          vselfBinsManagerElectro);
    }

  MPI_Barrier(MPI_COMM_WORLD);
  enowfc_time = MPI_Wtime() - enowfc_time;

  forcetotal_time = MPI_Wtime() - forcetotal_time;

  if (this_process == 0 && dftParameters::verbosity >= 4)
    std::cout
      << "Total time for configurational stress computation except Eself contribution: "
      << forcetotal_time << std::endl;

  if (dftParameters::verbosity >= 4)
    {
      pcout << " Time taken for initialization in stress: " << init_time
            << std::endl;
      pcout << " Time taken for non wfc in stress: " << enowfc_time
            << std::endl;
    }
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::computeStressEEshelbyEElectroPhiTot(
  const MatrixFree<3, double> &    matrixFreeDataElectro,
  const unsigned int               phiTotDofHandlerIndexElectro,
  const unsigned int               smearedChargeQuadratureId,
  const unsigned int               lpspQuadratureIdElectro,
  const distributedCPUVec<double> &phiTotRhoOutElectro,
  const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectro,
  const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectroLpsp,
  const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValuesElectro,
  const std::map<dealii::CellId, std::vector<double>>
    &gradRhoOutValuesElectroLpsp,
  const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
  const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
    &                                              pseudoVLocAtomsElectro,
  const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro)
{
  FEEvaluation<C_DIM,
               1,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               C_DIM>
    forceEvalElectro(matrixFreeDataElectro,
                     d_forceDofHandlerIndexElectro,
                     dftPtr->d_densityQuadratureId);

  FEEvaluation<C_DIM,
               FEOrderElectro,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               1>
    phiTotEvalElectro(matrixFreeDataElectro,
                      phiTotDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureId);

  FEEvaluation<C_DIM,
               FEOrderElectro,
               C_num1DQuadSmearedCharge() * C_numCopies1DQuadSmearedCharge(),
               1>
    phiTotEvalSmearedCharge(matrixFreeDataElectro,
                            phiTotDofHandlerIndexElectro,
                            smearedChargeQuadratureId);

  FEEvaluation<C_DIM,
               1,
               C_num1DQuadSmearedCharge() * C_numCopies1DQuadSmearedCharge(),
               C_DIM>
    forceEvalSmearedCharge(matrixFreeDataElectro,
                           d_forceDofHandlerIndexElectro,
                           smearedChargeQuadratureId);

  FEEvaluation<C_DIM,
               1,
               C_num1DQuadLPSP<FEOrderElectro>() * C_numCopies1DQuadLPSP(),
               C_DIM>
    forceEvalElectroLpsp(matrixFreeDataElectro,
                         d_forceDofHandlerIndexElectro,
                         lpspQuadratureIdElectro);

  FEValues<C_DIM> feVselfValuesElectro(
    matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro)
      .get_fe(),
    matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro),
    update_values | update_quadrature_points);

  const unsigned int numQuadPoints         = forceEvalElectro.n_q_points;
  const unsigned int numQuadPointsSmearedb = forceEvalSmearedCharge.n_q_points;
  const unsigned int numQuadPointsLpsp     = forceEvalElectroLpsp.n_q_points;

  if (gradRhoOutValuesElectroLpsp.size() != 0)
    AssertThrow(
      gradRhoOutValuesElectroLpsp.begin()->second.size() ==
        3 * numQuadPointsLpsp,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;


  Tensor<1, C_DIM, VectorizedArray<double>> zeroTensor;
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    {
      zeroTensor[idim] = make_vectorized_array(0.0);
    }

  Tensor<2, C_DIM, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    for (unsigned int jdim = 0; jdim < C_DIM; jdim++)
      zeroTensor2[idim][jdim] = make_vectorized_array(0.0);

  std::vector<VectorizedArray<double>> rhoQuadsElectro(
    numQuadPoints, make_vectorized_array(0.0));
  std::vector<VectorizedArray<double>> rhoQuadsElectroLpsp(
    numQuadPointsLpsp, make_vectorized_array(0.0));
  std::vector<VectorizedArray<double>> smearedbQuads(
    numQuadPointsSmearedb, make_vectorized_array(0.0));
  std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
                                                         gradPhiTotSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);
  std::vector<Tensor<1, C_DIM, VectorizedArray<double>>> gradRhoQuadsElectro(
    numQuadPoints, zeroTensor);
  std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
                                       gradRhoQuadsElectroLpsp(numQuadPointsLpsp, zeroTensor);
  std::vector<VectorizedArray<double>> pseudoVLocQuadsElectro(
    numQuadPointsLpsp, make_vectorized_array(0.0));
  for (unsigned int cell = 0; cell < matrixFreeDataElectro.n_macro_cells();
       ++cell)
    {
      std::set<unsigned int> nonTrivialSmearedChargeAtomImageIdsMacroCell;

      const unsigned int numSubCells =
        matrixFreeDataElectro.n_components_filled(cell);

      if (dftParameters::smearedNuclearCharges)
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr =
              matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
            dealii::CellId                   subCellId = subCellPtr->id();
            const std::vector<unsigned int> &temp =
              dftPtr->d_bCellNonTrivialAtomImageIds.find(subCellId)->second;
            for (int i = 0; i < temp.size(); i++)
              nonTrivialSmearedChargeAtomImageIdsMacroCell.insert(temp[i]);
          }

      forceEvalElectro.reinit(cell);
      forceEvalElectroLpsp.reinit(cell);

      phiTotEvalElectro.reinit(cell);
      phiTotEvalElectro.read_dof_values_plain(phiTotRhoOutElectro);
      phiTotEvalElectro.evaluate(true, true);

      if (dftParameters::smearedNuclearCharges)
        {
          forceEvalSmearedCharge.reinit(cell);
          phiTotEvalSmearedCharge.reinit(cell);
          phiTotEvalSmearedCharge.read_dof_values_plain(phiTotRhoOutElectro);
          phiTotEvalSmearedCharge.evaluate(false, true);
        }

      std::fill(rhoQuadsElectro.begin(),
                rhoQuadsElectro.end(),
                make_vectorized_array(0.0));
      std::fill(rhoQuadsElectroLpsp.begin(),
                rhoQuadsElectroLpsp.end(),
                make_vectorized_array(0.0));
      std::fill(gradRhoQuadsElectro.begin(),
                gradRhoQuadsElectro.end(),
                zeroTensor);
      std::fill(gradRhoQuadsElectroLpsp.begin(),
                gradRhoQuadsElectroLpsp.end(),
                zeroTensor);
      std::fill(pseudoVLocQuadsElectro.begin(),
                pseudoVLocQuadsElectro.end(),
                make_vectorized_array(0.0));
      std::fill(smearedbQuads.begin(),
                smearedbQuads.end(),
                make_vectorized_array(0.0));
      std::fill(gradPhiTotSmearedChargeQuads.begin(),
                gradPhiTotSmearedChargeQuads.end(),
                zeroTensor);

      double sum = 0.0;
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
          dealii::CellId subCellId = subCellPtr->id();
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            rhoQuadsElectro[q][iSubCell] =
              rhoOutValuesElectro.find(subCellId)->second[q];


          if (dftParameters::isPseudopotential ||
              dftParameters::smearedNuclearCharges)
            {
              const std::vector<double> &tempPseudoVal =
                pseudoVLocElectro.find(subCellId)->second;
              const std::vector<double> &tempLpspRhoVal =
                rhoOutValuesElectroLpsp.find(subCellId)->second;
              const std::vector<double> &tempLpspGradRhoVal =
                gradRhoOutValuesElectroLpsp.find(subCellId)->second;
              for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
                {
                  pseudoVLocQuadsElectro[q][iSubCell] = tempPseudoVal[q];
                  rhoQuadsElectroLpsp[q][iSubCell]    = tempLpspRhoVal[q];
                  gradRhoQuadsElectroLpsp[q][0][iSubCell] =
                    tempLpspGradRhoVal[3 * q + 0];
                  gradRhoQuadsElectroLpsp[q][1][iSubCell] =
                    tempLpspGradRhoVal[3 * q + 1];
                  gradRhoQuadsElectroLpsp[q][2][iSubCell] =
                    tempLpspGradRhoVal[3 * q + 2];
                }
            }

          if (dftParameters::smearedNuclearCharges)
            {
              const std::vector<double> &bQuadValuesCell =
                dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;
              for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                {
                  smearedbQuads[q][iSubCell] = bQuadValuesCell[q];
                  sum += bQuadValuesCell[q];
                }
            }
        }

      if (dftParameters::isPseudopotential ||
          dftParameters::smearedNuclearCharges)
        {
          addEPSPStressContribution(
            feVselfValuesElectro,
            forceEvalElectroLpsp,
            matrixFreeDataElectro,
            phiTotDofHandlerIndexElectro,
            cell,
            gradRhoQuadsElectroLpsp,
            pseudoVLocAtomsElectro,
            vselfBinsManagerElectro,
            d_cellsVselfBallsClosestAtomIdDofHandlerElectro);
        }

      Tensor<2, C_DIM, VectorizedArray<double>> EQuadSum = zeroTensor2;
      for (unsigned int q = 0; q < numQuadPoints; ++q)
        {
          VectorizedArray<double> phiTotElectro_q =
            phiTotEvalElectro.get_value(q);
          VectorizedArray<double> phiExtElectro_q = make_vectorized_array(0.0);
          Tensor<1, C_DIM, VectorizedArray<double>> gradPhiTotElectro_q =
            phiTotEvalElectro.get_gradient(q);

          Tensor<2, C_DIM, VectorizedArray<double>> E =
            eshelbyTensor::getEElectroEshelbyTensor(phiTotElectro_q,
                                                    gradPhiTotElectro_q,
                                                    rhoQuadsElectro[q]);

          EQuadSum += E * forceEvalElectro.JxW(q);
        }

      if (dftParameters::isPseudopotential ||
          dftParameters::smearedNuclearCharges)
        for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
          {
            VectorizedArray<double> phiExtElectro_q =
              make_vectorized_array(0.0);
            Tensor<2, C_DIM, VectorizedArray<double>> E = zeroTensor2;

            EQuadSum += E * forceEvalElectroLpsp.JxW(q);
          }

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < C_DIM; ++idim)
          for (unsigned int jdim = 0; jdim < C_DIM; ++jdim)
            d_stress[idim][jdim] += EQuadSum[idim][jdim][iSubCell];

      if (dftParameters::smearedNuclearCharges && std::abs(sum) > 1e-9)
        {
          for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
            {
              gradPhiTotSmearedChargeQuads[q] =
                phiTotEvalSmearedCharge.get_gradient(q);
            }

          addEPhiTotSmearedStressContribution(
            forceEvalSmearedCharge,
            matrixFreeDataElectro,
            cell,
            gradPhiTotSmearedChargeQuads,
            std::vector<unsigned int>(
              nonTrivialSmearedChargeAtomImageIdsMacroCell.begin(),
              nonTrivialSmearedChargeAtomImageIdsMacroCell.end()),
            dftPtr->d_bQuadAtomIdsAllAtomsImages,
            smearedbQuads);
        }

    } // cell loop
}
