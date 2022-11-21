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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

//
// Initialize rho by reading in single-atom electron-density and fit a spline
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::initLocalPseudoPotential(
  const DoFHandler<3> &                    _dofHandler,
  const unsigned int                       lpspQuadratureId,
  const dealii::MatrixFree<3, double> &    _matrix_free_data,
  const unsigned int                       _phiExtDofHandlerIndex,
  const dealii::AffineConstraints<double> &_phiExtConstraintMatrix,
  const std::map<dealii::types::global_dof_index, Point<3>> &_supportPoints,
  const vselfBinsManager<FEOrder, FEOrderElectro> &          vselfBinManager,
  distributedCPUVec<double> &                                phiExt,
  std::map<dealii::CellId, std::vector<double>> &            _pseudoValues,
  std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
    &_pseudoValuesAtoms)
{
  _pseudoValues.clear();
  _pseudoValuesAtoms.clear();

  //
  // Reading single atom rho initial guess
  //
  std::map<unsigned int, alglib::spline1dinterpolant>      pseudoSpline;
  std::map<unsigned int, std::vector<std::vector<double>>> pseudoPotentialData;
  std::map<unsigned int, double>                           outerMostDataPoint;
  // FIXME: the truncation tolerance can potentially be loosened
  // further for production runs where more accurate meshes are used
  const double truncationTol =
    d_dftParamsPtr->reproducible_output ? 1.0e-8 : 1.0e-7;
  const double maxAllowedTail = 8.0001;
  double       maxTail        = 0.0;
  if (d_dftParamsPtr->isPseudopotential)
    {
      //
      // loop over atom types
      //
      for (std::set<unsigned int>::iterator it = atomTypes.begin();
           it != atomTypes.end();
           it++)
        {
          char pseudoFile[256];

          strcpy(pseudoFile,
                 (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                  "/locPot.dat")
                   .c_str());

          dftUtils::readFile(2, pseudoPotentialData[*it], pseudoFile);
          unsigned int        numRows = pseudoPotentialData[*it].size() - 1;
          std::vector<double> xData(numRows), yData(numRows);

          unsigned int maxRowId = 0;
          for (unsigned int irow = 0; irow < numRows; ++irow)
            {
              xData[irow] = pseudoPotentialData[*it][irow][0];
              yData[irow] = pseudoPotentialData[*it][irow][1];

              if (irow > 0 && xData[irow] < maxAllowedTail)
                {
                  if (std::abs(yData[irow] -
                               (-((double)d_atomTypeAtributes[*it]) /
                                xData[irow])) > truncationTol)
                    maxRowId = irow;
                }
            }

          // interpolate pseudopotentials
          alglib::real_1d_array x;
          x.setcontent(numRows, &xData[0]);
          alglib::real_1d_array y;
          y.setcontent(numRows, &yData[0]);
          alglib::ae_int_t bound_type_l = 0;
          alglib::ae_int_t bound_type_r = 1;
          const double     slopeL =
            (pseudoPotentialData[*it][1][1] - pseudoPotentialData[*it][0][1]) /
            (pseudoPotentialData[*it][1][0] - pseudoPotentialData[*it][0][0]);
          const double slopeR = -pseudoPotentialData[*it][numRows - 1][1] /
                                pseudoPotentialData[*it][numRows - 1][0];
          spline1dbuildcubic(x,
                             y,
                             numRows,
                             bound_type_l,
                             slopeL,
                             bound_type_r,
                             slopeR,
                             pseudoSpline[*it]);
          outerMostDataPoint[*it] = xData[maxRowId];

          if (outerMostDataPoint[*it] > maxTail)
            maxTail = outerMostDataPoint[*it];
        }
    }
  else
    {
      maxTail = maxAllowedTail;
      for (std::set<unsigned int>::iterator it = atomTypes.begin();
           it != atomTypes.end();
           it++)
        outerMostDataPoint[*it] = maxAllowedTail;
    }

  const double cutOffForPsp =
    std::max(vselfBinManager.getStoredAdaptiveBallRadius() + 6.0, maxTail);

  //
  // Initialize pseudopotential
  //
  FEValues<3>        fe_values(_dofHandler.get_fe(),
                        _matrix_free_data.get_quadrature(lpspQuadratureId),
                        update_quadrature_points);
  const unsigned int n_q_points =
    _matrix_free_data.get_quadrature(lpspQuadratureId).size();

  const int numberGlobalCharges = atomLocations.size();
  //
  // get number of image charges used only for periodic
  //
  const int numberImageCharges = d_imageIds.size();

  // distributedCPUVec<double> phiExt;
  //_matrix_free_data.initialize_dof_vector(phiExt,_phiExtDofHandlerIndex);
  phiExt = 0;

  double init_1;
  MPI_Barrier(d_mpiCommParent);
  init_1 = MPI_Wtime();

  const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
    &                partitioner = phiExt.get_partitioner();
  const unsigned int localSize   = partitioner->local_size();
  const unsigned int n_ghosts    = partitioner->n_ghost_indices();
  const unsigned int totalSize   = localSize + n_ghosts;


  const std::vector<std::map<dealii::types::global_dof_index, int>>
    &boundaryNodeMapBinsOnlyChargeId =
      vselfBinManager.getBoundaryFlagsBinsOnlyChargeId();
  const std::vector<std::map<dealii::types::global_dof_index, dealii::Point<3>>>
    &dofClosestChargeLocationMapBins =
      vselfBinManager.getClosestAtomLocationsBins();
  const std::map<unsigned int, unsigned int> &atomIdBinIdMap =
    vselfBinManager.getAtomIdBinIdMapLocalAllImages();

  const unsigned int dofs_per_cell = _dofHandler.get_fe().dofs_per_cell;
  DoFHandler<3>::active_cell_iterator subCellPtr;

  dealii::BoundingBox<3> boundingBoxTria(
    vectorTools::createBoundingBoxTriaLocallyOwned(_dofHandler));
  dealii::Tensor<1, 3, double> tempDisp;
  tempDisp[0] = cutOffForPsp;
  tempDisp[1] = cutOffForPsp;
  tempDisp[2] = cutOffForPsp;
  std::pair<dealii::Point<3, double>, dealii::Point<3, double>> boundaryPoints;

  std::vector<double> atomsImagesPositions(
    (numberGlobalCharges + numberImageCharges) * 3);
  std::vector<double> atomsImagesCharges(
    (numberGlobalCharges + numberImageCharges));
  for (unsigned int iAtom = 0; iAtom < numberGlobalCharges + numberImageCharges;
       iAtom++)
    {
      if (iAtom < numberGlobalCharges)
        {
          atomsImagesPositions[iAtom * 3 + 0] = atomLocations[iAtom][2];
          atomsImagesPositions[iAtom * 3 + 1] = atomLocations[iAtom][3];
          atomsImagesPositions[iAtom * 3 + 2] = atomLocations[iAtom][4];
          if (d_dftParamsPtr->isPseudopotential)
            atomsImagesCharges[iAtom] = atomLocations[iAtom][1];
          else
            atomsImagesCharges[iAtom] = atomLocations[iAtom][0];
        }
      else
        {
          const unsigned int iImageCharge = iAtom - numberGlobalCharges;
          atomsImagesPositions[iAtom * 3 + 0] =
            d_imagePositions[iImageCharge][0];
          atomsImagesPositions[iAtom * 3 + 1] =
            d_imagePositions[iImageCharge][1];
          atomsImagesPositions[iAtom * 3 + 2] =
            d_imagePositions[iImageCharge][2];
          if (d_dftParamsPtr->isPseudopotential)
            atomsImagesCharges[iAtom] =
              atomLocations[d_imageIds[iImageCharge]][1];
          else
            atomsImagesCharges[iAtom] =
              atomLocations[d_imageIds[iImageCharge]][0];
        }
    }

  typename DoFHandler<3>::active_cell_iterator cell =
                                                 _dofHandler.begin_active(),
                                               endc     = _dofHandler.end();
  int                                    numberElements = 0;
  std::map<dealii::CellId, unsigned int> cellIdToElemIdMap;
  std::map<unsigned int, dealii::CellId> elemIdToCellIdMap;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        cellIdToElemIdMap[cell->id()]     = numberElements;
        elemIdToCellIdMap[numberElements] = cell->id();

        std::vector<double> &pseudoVLoc = _pseudoValues[cell->id()];
        pseudoVLoc.resize(n_q_points, 0.0);
        numberElements++;
      }

  const int numberDofs = phiExt.local_size();
  // kpoint group parallelization data structures
  const unsigned int numberKptGroups =
    dealii::Utilities::MPI::n_mpi_processes(interpoolcomm);

  const unsigned int kptGroupTaskId =
    dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
  std::vector<int> kptGroupLowHighPlusOneIndicesStep1;

  if (numberDofs > 0)
    dftUtils::createKpointParallelizationIndices(
      interpoolcomm, numberDofs, kptGroupLowHighPlusOneIndicesStep1);

  for (unsigned int localDofId = 0; localDofId < phiExt.local_size();
       ++localDofId)
    {
      if (localDofId <
            kptGroupLowHighPlusOneIndicesStep1[2 * kptGroupTaskId + 1] &&
          localDofId >= kptGroupLowHighPlusOneIndicesStep1[2 * kptGroupTaskId])
        {
          const dealii::types::global_dof_index dofId =
            partitioner->local_to_global(localDofId);
          const Point<3> &nodalCoor = _supportPoints.find(dofId)->second;
          if (!_phiExtConstraintMatrix.is_constrained(dofId))
            {
              Point<3> atom;
              double   atomCharge;
              int      atomicNumber;
              int      chargeId;
              double   distanceToAtom;
              double   sumVal = 0.0;
              double   val;
              double   diffx;
              double   diffy;
              double   diffz;
              for (unsigned int iAtom = 0;
                   iAtom < (atomLocations.size() + numberImageCharges);
                   ++iAtom)
                {
                  diffx = nodalCoor[0] - atomsImagesPositions[iAtom * 3 + 0];
                  diffy = nodalCoor[1] - atomsImagesPositions[iAtom * 3 + 1];
                  diffz = nodalCoor[2] - atomsImagesPositions[iAtom * 3 + 2];
                  atomCharge = atomsImagesCharges[iAtom];

                  distanceToAtom =
                    std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);

                  if (distanceToAtom < cutOffForPsp)
                    {
                      if (iAtom < numberGlobalCharges)
                        {
                          chargeId = iAtom;
                        }
                      else
                        {
                          const unsigned int iImageCharge =
                            iAtom - numberGlobalCharges;
                          chargeId = d_imageIds[iImageCharge];
                        }

                      if (atomIdBinIdMap.find(chargeId) != atomIdBinIdMap.end())
                        {
                          const unsigned int binId =
                            atomIdBinIdMap.find(chargeId)->second;
                          const int boundaryFlagChargeId =
                            boundaryNodeMapBinsOnlyChargeId[binId]
                              .find(dofId)
                              ->second;

                          if (boundaryFlagChargeId == chargeId)
                            {
                              atom[0] = atomsImagesPositions[iAtom * 3 + 0];
                              atom[1] = atomsImagesPositions[iAtom * 3 + 1];
                              atom[2] = atomsImagesPositions[iAtom * 3 + 2];

                              if (dofClosestChargeLocationMapBins[binId]
                                    .find(dofId)
                                    ->second.distance(atom) < 1e-5)
                                {
                                  const distributedCPUVec<double> &vselfBin =
                                    vselfBinManager.getVselfFieldBins()[binId];
                                  val = vselfBin.local_element(localDofId);
                                }
                              else
                                val = -atomCharge / distanceToAtom;
                            }
                          else
                            val = -atomCharge / distanceToAtom;
                        }
                    }
                  else
                    {
                      val = -atomCharge / distanceToAtom;
                    }

                  sumVal += val;
                }
              phiExt.local_element(localDofId) = sumVal;
            }
        } // interpool comm parallelization
    }     // dof loop

  if (numberDofs > 0 && numberKptGroups > 1)
    MPI_Allreduce(MPI_IN_PLACE,
                  phiExt.begin(),
                  numberDofs,
                  MPI_DOUBLE,
                  MPI_SUM,
                  interpoolcomm);
  MPI_Barrier(interpoolcomm);

  _phiExtConstraintMatrix.distribute(phiExt);
  phiExt.update_ghost_values();
  // pcout<<"L2 Norm Value of phiext: "<<phiExt.l2_norm()<<std::endl;

  MPI_Barrier(d_mpiCommParent);
  init_1 = MPI_Wtime() - init_1;
  if (d_dftParamsPtr->verbosity >= 4)
    pcout << "initLocalPSP: Time taken for init1: " << init_1 << std::endl;

  double init_2;
  MPI_Barrier(d_mpiCommParent);
  init_2 = MPI_Wtime();

  FEEvaluation<3,
               FEOrderElectro,
               C_num1DQuadLPSP<FEOrderElectro>() * C_numCopies1DQuadLPSP()>
    feEvalObj(_matrix_free_data, _phiExtDofHandlerIndex, lpspQuadratureId);
  AssertThrow(
    _matrix_free_data.get_quadrature(lpspQuadratureId).size() ==
      feEvalObj.n_q_points,
    dealii::ExcMessage(
      "DFT-FE Error: mismatch in quadrature rule usage in initLocalPseudoPotential."));

  const int numMacroCells = _matrix_free_data.n_macro_cells();

  std::vector<int> kptGroupLowHighPlusOneIndicesStep2;

  if (numMacroCells > 0)
    dftUtils::createKpointParallelizationIndices(
      interpoolcomm, numMacroCells, kptGroupLowHighPlusOneIndicesStep2);

  for (unsigned int macrocell = 0;
       macrocell < _matrix_free_data.n_macro_cells();
       ++macrocell)
    {
      if (macrocell <
            kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId + 1] &&
          macrocell >= kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId])
        {
          feEvalObj.reinit(macrocell);
          feEvalObj.read_dof_values(phiExt);
          feEvalObj.evaluate(true, false);

          Point<3> atom;
          int      atomicNumber;
          double   atomCharge;


          for (unsigned int iSubCell = 0;
               iSubCell < _matrix_free_data.n_components_filled(macrocell);
               ++iSubCell)
            {
              subCellPtr =
                _matrix_free_data.get_cell_iterator(macrocell,
                                                    iSubCell,
                                                    _phiExtDofHandlerIndex);
              dealii::CellId subCellId = subCellPtr->id();

              std::vector<double> &pseudoVLoc = _pseudoValues[subCellId];

              Point<3> quadPoint;
              double   value, distanceToAtom, distanceToAtomInv;

              fe_values.reinit(subCellPtr);


              // loop over quad points
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const Point<3> &quadPoint = fe_values.quadrature_point(q);

                  double temp;
                  double tempVal = 0.0;
                  double diffx;
                  double diffy;
                  double diffz;
                  // loop over atoms
                  for (unsigned int iAtom = 0;
                       iAtom < numberGlobalCharges + numberImageCharges;
                       iAtom++)
                    {
                      diffx =
                        quadPoint[0] - atomsImagesPositions[iAtom * 3 + 0];
                      diffy =
                        quadPoint[1] - atomsImagesPositions[iAtom * 3 + 1];
                      diffz =
                        quadPoint[2] - atomsImagesPositions[iAtom * 3 + 2];

                      atomCharge = atomsImagesCharges[iAtom];

                      distanceToAtom = std::sqrt(diffx * diffx + diffy * diffy +
                                                 diffz * diffz);
                      distanceToAtomInv = 1.0 / distanceToAtom;

                      if (distanceToAtom <= maxTail)
                        {
                          if (iAtom < numberGlobalCharges)
                            {
                              atomicNumber =
                                std::round(atomLocations[iAtom][0]);
                            }
                          else
                            {
                              const unsigned int iImageCharge =
                                iAtom - numberGlobalCharges;
                              atomicNumber = std::round(
                                atomLocations[d_imageIds[iImageCharge]][0]);
                            }

                          if (distanceToAtom <=
                              outerMostDataPoint[atomicNumber])
                            {
                              if (d_dftParamsPtr->isPseudopotential)
                                {
                                  value = alglib::spline1dcalc(
                                    pseudoSpline[atomicNumber], distanceToAtom);
                                }
                              else
                                {
                                  value = -atomCharge * distanceToAtomInv;
                                }
                            }
                          else
                            {
                              value = -atomCharge * distanceToAtomInv;
                            }
                        }
                      else
                        {
                          value = -atomCharge * distanceToAtomInv;
                        }
                      tempVal += value;
                    } // atom loop
                  pseudoVLoc[q] = tempVal;
                } // quad loop
            }     // subcell loop

          for (unsigned int iSubCell = 0;
               iSubCell < _matrix_free_data.n_components_filled(macrocell);
               ++iSubCell)
            {
              subCellPtr =
                _matrix_free_data.get_cell_iterator(macrocell,
                                                    iSubCell,
                                                    _phiExtDofHandlerIndex);
              dealii::CellId       subCellId  = subCellPtr->id();
              std::vector<double> &pseudoVLoc = _pseudoValues[subCellId];
              // loop over quad points
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  pseudoVLoc[q] -= feEvalObj.get_value(q)[iSubCell];
                } // loop over quad points
            }     // subcell loop
        }         // intercomm paral
    }             // cell loop

  if (numMacroCells > 0 && numberKptGroups > 1)
    {
      std::vector<double> tempPseudoValuesFlattened(numberElements * n_q_points,
                                                    0.0);

      cell = _dofHandler.begin_active();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            const unsigned int   elemId     = cellIdToElemIdMap[cell->id()];
            std::vector<double> &pseudoVLoc = _pseudoValues[cell->id()];
            for (unsigned int q = 0; q < n_q_points; ++q)
              tempPseudoValuesFlattened[elemId * n_q_points + q] =
                pseudoVLoc[q];
          }

      MPI_Allreduce(MPI_IN_PLACE,
                    &tempPseudoValuesFlattened[0],
                    numberElements * n_q_points,
                    MPI_DOUBLE,
                    MPI_SUM,
                    interpoolcomm);
      MPI_Barrier(interpoolcomm);

      cell = _dofHandler.begin_active();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            const unsigned int   elemId     = cellIdToElemIdMap[cell->id()];
            std::vector<double> &pseudoVLoc = _pseudoValues[cell->id()];
            for (unsigned int q = 0; q < n_q_points; ++q)
              pseudoVLoc[q] =
                tempPseudoValuesFlattened[elemId * n_q_points + q];
          }
    }


  MPI_Barrier(d_mpiCommParent);
  init_2 = MPI_Wtime() - init_2;
  if (d_dftParamsPtr->verbosity >= 4)
    pcout << "initLocalPSP: Time taken for init2: " << init_2 << std::endl;

  double init_3;
  MPI_Barrier(d_mpiCommParent);
  init_3 = MPI_Wtime();

  std::vector<int> kptGroupLowHighPlusOneIndicesStep3;

  if (numberElements > 0)
    dftUtils::createKpointParallelizationIndices(
      interpoolcomm, numberElements, kptGroupLowHighPlusOneIndicesStep3);

  std::vector<double> pseudoVLocAtom(n_q_points);
  unsigned int        ielem = 0;
  cell                      = _dofHandler.begin_active();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          if ((ielem <
                 kptGroupLowHighPlusOneIndicesStep3[2 * kptGroupTaskId + 1] &&
               ielem >= kptGroupLowHighPlusOneIndicesStep3[2 * kptGroupTaskId]))
            {
              // compute values for the current elements
              fe_values.reinit(cell);

              Point<3> atom;
              int      atomicNumber;
              double   atomCharge;

              // loop over atoms
              for (unsigned int iAtom = 0;
                   iAtom < numberGlobalCharges + d_imagePositionsTrunc.size();
                   iAtom++)
                {
                  if (iAtom < numberGlobalCharges)
                    {
                      atom[0] = atomLocations[iAtom][2];
                      atom[1] = atomLocations[iAtom][3];
                      atom[2] = atomLocations[iAtom][4];
                      if (d_dftParamsPtr->isPseudopotential)
                        atomCharge = atomLocations[iAtom][1];
                      else
                        atomCharge = atomLocations[iAtom][0];
                      atomicNumber = std::round(atomLocations[iAtom][0]);
                    }
                  else
                    {
                      const unsigned int iImageCharge =
                        iAtom - numberGlobalCharges;
                      atom[0] = d_imagePositionsTrunc[iImageCharge][0];
                      atom[1] = d_imagePositionsTrunc[iImageCharge][1];
                      atom[2] = d_imagePositionsTrunc[iImageCharge][2];
                      if (d_dftParamsPtr->isPseudopotential)
                        atomCharge =
                          atomLocations[d_imageIdsTrunc[iImageCharge]][1];
                      else
                        atomCharge =
                          atomLocations[d_imageIdsTrunc[iImageCharge]][0];
                      atomicNumber = std::round(
                        atomLocations[d_imageIdsTrunc[iImageCharge]][0]);
                    }


                  boundaryPoints.first  = atom - tempDisp;
                  boundaryPoints.second = atom + tempDisp;
                  dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

                  if (boundingBoxTria.get_neighbor_type(
                        boundingBoxAroundAtom) == NeighborType::not_neighbors)
                    continue;

                  bool         isPseudoDataInCell = false;
                  Point<3>     quadPoint;
                  double       value, distanceToAtom;
                  const double cutoff = outerMostDataPoint[atomicNumber];
                  // loop over quad points
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                      const Point<3> &quadPoint = fe_values.quadrature_point(q);
                      distanceToAtom            = quadPoint.distance(atom);
                      if (distanceToAtom <= cutoff)
                        {
                          if (d_dftParamsPtr->isPseudopotential)
                            {
                              value =
                                alglib::spline1dcalc(pseudoSpline[atomicNumber],
                                                     distanceToAtom);
                            }
                          else
                            {
                              value = -atomCharge / distanceToAtom;
                            }
                        }
                      else
                        {
                          value = -atomCharge / distanceToAtom;
                        }

                      if (distanceToAtom <= cutOffForPsp)
                        isPseudoDataInCell = true;

                      pseudoVLocAtom[q] = value;
                    } // loop over quad points
                  if (isPseudoDataInCell)
                    {
                      _pseudoValuesAtoms[iAtom][cell->id()] = pseudoVLocAtom;
                    }
                } // loop over atoms
            }     // kpt paral loop
          ielem++;
        } // cell locally owned check
    }     // cell loop

  if (numberElements > 0 && numberKptGroups > 1)
    {
      // arranged as iAtom, elemid, and quad data
      std::vector<double> sendData;
      int                 sendCount = 0;
      // loop over atoms
      for (unsigned int iAtom = 0;
           iAtom < numberGlobalCharges + d_imagePositionsTrunc.size();
           iAtom++)
        {
          if (_pseudoValuesAtoms.find(iAtom) != _pseudoValuesAtoms.end())
            {
              cell = _dofHandler.begin_active();
              for (; cell != endc; ++cell)
                if (cell->is_locally_owned())
                  {
                    if (_pseudoValuesAtoms[iAtom].find(cell->id()) !=
                        _pseudoValuesAtoms[iAtom].end())
                      {
                        sendCount++;
                        pseudoVLocAtom = _pseudoValuesAtoms[iAtom][cell->id()];
                        sendData.push_back(iAtom);
                        sendData.push_back(cellIdToElemIdMap[cell->id()]);
                        sendData.insert(sendData.end(),
                                        pseudoVLocAtom.begin(),
                                        pseudoVLocAtom.end());
                      }
                  } // cell locally owned loop
            }
        } // iatom loop

      sendCount = sendCount * (2 + n_q_points);

      if (sendCount == 0)
        {
          sendCount = (2 + n_q_points);
          sendData.resize(sendCount, 0);
          sendData[0] = -1;
        }

      std::vector<int> recvCounts(numberKptGroups, 0);
      int              ierr = MPI_Allgather(
        &sendCount, 1, MPI_INT, &recvCounts[0], 1, MPI_INT, interpoolcomm);

      if (ierr)
        AssertThrow(false,
                    dealii::ExcMessage(
                      "DFT-FE Error: MPI Error in init local psp"));


      const int recvDataSize =
        std::accumulate(recvCounts.begin(), recvCounts.end(), 0);


      std::vector<int> displacements(numberKptGroups, 0);
      int              disp = 0;
      for (int i = 0; i < numberKptGroups; ++i)
        {
          displacements[i] = disp;
          disp += recvCounts[i];
        }

      std::vector<double> recvData(recvDataSize, 0.0);

      ierr = MPI_Allgatherv(&sendData[0],
                            sendCount,
                            MPI_DOUBLE,
                            &recvData[0],
                            &recvCounts[0],
                            &displacements[0],
                            MPI_DOUBLE,
                            interpoolcomm);

      if (ierr)
        AssertThrow(false,
                    dealii::ExcMessage(
                      "DFT-FE Error: MPI Error in init local psp"));


      for (unsigned int i = 0; i < recvDataSize / (2 + n_q_points); i++)
        {
          const int iatom = std::round(recvData[i * (2 + n_q_points) + 0]);
          const unsigned int elementId =
            std::round(recvData[i * (2 + n_q_points) + 1]);


          if (iatom != -1)
            {
              const dealii::CellId writeCellId = elemIdToCellIdMap[elementId];
              if (_pseudoValuesAtoms[iatom].find(writeCellId) ==
                  _pseudoValuesAtoms[iatom].end())
                {
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    pseudoVLocAtom[q] = recvData[i * (2 + n_q_points) + 2 + q];

                  _pseudoValuesAtoms[iatom][writeCellId] = pseudoVLocAtom;
                }
            }
        }

      MPI_Barrier(interpoolcomm);
    }

  MPI_Barrier(d_mpiCommParent);
  init_3 = MPI_Wtime() - init_3;
  if (d_dftParamsPtr->verbosity >= 4)
    pcout << "initLocalPSP: Time taken for init3: " << init_3 << std::endl;
}
