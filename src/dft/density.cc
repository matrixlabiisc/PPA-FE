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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh, Sambit Das
//

//source file for electron density related computations

  template<unsigned int FEOrder>
void dftClass<FEOrder>::popOutRhoInRhoOutVals()
{

  //pop out rhoInVals and rhoOutVals if their size exceeds mixing history size

  if(dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
  {
    if(d_rhoInNodalVals.size() == dftParameters::mixingHistory)
    {
      d_rhoInNodalVals.pop_front();
      d_rhoOutNodalVals.pop_front();
    }
  }
  else
  {
    //pop out rhoInVals and rhoOutVals if their size exceeds mixing history size
    if(rhoInVals.size() == dftParameters::mixingHistory)
    {
      rhoInVals.pop_front();
      rhoOutVals.pop_front();

      if(dftParameters::spinPolarized==1)
      {
        rhoInValsSpinPolarized.pop_front();
        rhoOutValsSpinPolarized.pop_front();
      }

      if(dftParameters::xc_id == 4)//GGA
      {
        gradRhoInVals.pop_front();
        gradRhoOutVals.pop_front();
      }

      if(dftParameters::spinPolarized==1 && dftParameters::xc_id==4)
      {
        gradRhoInValsSpinPolarized.pop_front();
        gradRhoOutValsSpinPolarized.pop_front();
      }

      if (dftParameters::mixingMethod=="BROYDEN")
      {
        dFBroyden.pop_front();
        uBroyden.pop_front();
        if(dftParameters::xc_id == 4)//GGA
        {
          graddFBroyden.pop_front();
          gradUBroyden.pop_front();
        }
      }

    }

  }
}


//calculate electron density
  template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_rhoOut(
#ifdef DFTFE_WITH_GPU
    kohnShamDFTOperatorCUDAClass<FEOrder> & kohnShamDFTEigenOperator,
#endif
    const bool isConsiderSpectrumSplitting,
    const bool isGroundState)
{

  if(dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
  {
#ifdef DFTFE_WITH_GPU
    computeRhoNodalFromPSI(kohnShamDFTEigenOperator,isConsiderSpectrumSplitting);
#else
    computeRhoNodalFromPSI(isConsiderSpectrumSplitting);
#endif
    d_rhoOutNodalValues.update_ghost_values();
    d_rhoOutNodalVals.push_back(d_rhoOutNodalValues);


    //interpolate nodal rhoOut data to quadrature data
    interpolateNodalDataToQuadratureData(d_matrixFreeDataPRefined,
        d_rhoOutNodalValues,
        *rhoOutValues,
        *gradRhoOutValues,
        *gradRhoOutValues,
        dftParameters::xc_id == 4);


    if(dftParameters::verbosity>=3)
    {
      pcout<<"Total Charge using nodal Rho out: "<< totalCharge(d_matrixFreeDataPRefined,d_rhoOutNodalValues)<<std::endl;
    }

  }
  else
  {

    resizeAndAllocateRhoTableStorage
      (rhoOutVals,
       gradRhoOutVals,
       rhoOutValsSpinPolarized,
       gradRhoOutValsSpinPolarized);

    rhoOutValues = &(rhoOutVals.back());
    if (dftParameters::spinPolarized==1)
      rhoOutValuesSpinPolarized = &(rhoOutValsSpinPolarized.back());

    if(dftParameters::xc_id == 4)
    {
      gradRhoOutValues = &(gradRhoOutVals.back());
      if (dftParameters::spinPolarized==1)
        gradRhoOutValuesSpinPolarized = &(gradRhoOutValsSpinPolarized.back());
    }

    DensityCalculator<FEOrder> densityCalculator;

#ifdef DFTFE_WITH_GPU
    if (dftParameters::useGPU)
      CUDA::computeRhoFromPSI(
          d_eigenVectorsFlattenedCUDA.begin(),
          d_eigenVectorsRotFracFlattenedCUDA.begin(),
          d_numEigenValues,
          d_numEigenValuesRR,
          d_eigenVectorsFlattenedSTL[0].size()/d_numEigenValues,
          eigenValues,
          fermiEnergy,
          fermiEnergyUp,
          fermiEnergyDown,
          kohnShamDFTEigenOperator,
          dofHandler,
          matrix_free_data.n_physical_cells(),
          matrix_free_data.get_dofs_per_cell(),
          QGauss<3>(C_num1DQuad<FEOrder>()).size(),
          d_kPointWeights,
          rhoOutValues,
          gradRhoOutValues,
          rhoOutValuesSpinPolarized,
          gradRhoOutValuesSpinPolarized,
          dftParameters::xc_id == 4,
          interpoolcomm,
          interBandGroupComm,
          isConsiderSpectrumSplitting && d_numEigenValues!=d_numEigenValuesRR);

    else
      densityCalculator.computeRhoFromPSI(
          d_eigenVectorsFlattenedSTL,
          d_eigenVectorsRotFracDensityFlattenedSTL,
          d_numEigenValues,
          d_numEigenValuesRR,
          eigenValues,
          fermiEnergy,
          fermiEnergyUp,
          fermiEnergyDown,
          dofHandler,
          constraintsNone,
          matrix_free_data,
          eigenDofHandlerIndex,
          0,
          localProc_dof_indicesReal,
          localProc_dof_indicesImag,           
          d_kPointWeights,
          rhoOutValues,
          gradRhoOutValues,
          rhoOutValuesSpinPolarized,
          gradRhoOutValuesSpinPolarized,
          dftParameters::xc_id == 4,
          interpoolcomm,
          interBandGroupComm,
          isConsiderSpectrumSplitting,
          false);
#else
    densityCalculator.computeRhoFromPSI(
        d_eigenVectorsFlattenedSTL,
        d_eigenVectorsRotFracDensityFlattenedSTL,
        d_numEigenValues,
        d_numEigenValuesRR,
        eigenValues,
        fermiEnergy,
        fermiEnergyUp,
        fermiEnergyDown,
        dofHandler,
        constraintsNone,
        matrix_free_data,
        eigenDofHandlerIndex,
        0,
        localProc_dof_indicesReal,
        localProc_dof_indicesImag,         
        d_kPointWeights,
        rhoOutValues,
        gradRhoOutValues,
        rhoOutValuesSpinPolarized,
        gradRhoOutValuesSpinPolarized,
        dftParameters::xc_id == 4,
        interpoolcomm,
        interBandGroupComm,
        isConsiderSpectrumSplitting,
        false);
#endif

    if (isGroundState && dftParameters::mixingMethod!="ANDERSON_WITH_KERKER")
    {
#ifdef DFTFE_WITH_GPU
      computeRhoNodalFromPSI(kohnShamDFTEigenOperator,isConsiderSpectrumSplitting);
#else
      computeRhoNodalFromPSI(isConsiderSpectrumSplitting);
#endif
      d_rhoOutNodalValues.update_ghost_values();
    }
  }

  popOutRhoInRhoOutVals();

  if (isGroundState && (dftParameters::isIonOpt || dftParameters::isCellOpt))
  {
    d_rhoOutNodalValuesSplit=d_rhoOutNodalValues;
    d_rhoOutNodalValuesSplit-=d_atomicRho;

    d_rhoOutNodalValuesSplit.update_ghost_values();
  }
}




template<unsigned int FEOrder>
  void dftClass<FEOrder>::resizeAndAllocateRhoTableStorage
(std::deque<std::map<dealii::CellId,std::vector<double> >> & rhoVals,
 std::deque<std::map<dealii::CellId,std::vector<double> >> & gradRhoVals,
 std::deque<std::map<dealii::CellId,std::vector<double> >> & rhoValsSpinPolarized,
 std::deque<std::map<dealii::CellId,std::vector<double> >> & gradRhoValsSpinPolarized)
{
  const unsigned int numQuadPoints = matrix_free_data.get_n_q_points(0);;

  //create new rhoValue tables
  rhoVals.push_back(std::map<dealii::CellId,std::vector<double> > ());
  if (dftParameters::spinPolarized==1)
    rhoValsSpinPolarized.push_back(std::map<dealii::CellId,std::vector<double> > ());

  if(dftParameters::xc_id == 4)
  {
    gradRhoVals.push_back(std::map<dealii::CellId, std::vector<double> >());
    if (dftParameters::spinPolarized==1)
      gradRhoValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
  }


  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      const dealii::CellId cellId=cell->id();
      rhoVals.back()[cellId] = std::vector<double>(numQuadPoints,0.0);
      if(dftParameters::xc_id == 4)
        gradRhoVals.back()[cellId] = std::vector<double>(3*numQuadPoints,0.0);

      if (dftParameters::spinPolarized==1)
      {
        rhoValsSpinPolarized.back()[cellId] = std::vector<double>(2*numQuadPoints,0.0);
        if(dftParameters::xc_id == 4)
          gradRhoValsSpinPolarized.back()[cellId]
            = std::vector<double>(6*numQuadPoints,0.0);
      }
    }
}


//rho data reinitilization without remeshing. The rho out of last ground state solve is made the rho in of the new solve
  template<unsigned int FEOrder>
void dftClass<FEOrder>::noRemeshRhoDataInit()
{

  if (rhoOutVals.size()>0 || d_rhoInNodalVals.size()>0)
  { 
    //create temporary copies of rho Out data
    std::map<dealii::CellId, std::vector<double> > rhoOutValuesCopy=*(rhoOutValues);

    std::map<dealii::CellId, std::vector<double> > gradRhoOutValuesCopy;
    if (dftParameters::xc_id==4)
    {
      gradRhoOutValuesCopy=*(gradRhoOutValues);
    }

    std::map<dealii::CellId, std::vector<double> > rhoOutValuesSpinPolarizedCopy;
    if(dftParameters::spinPolarized==1)
    {
      rhoOutValuesSpinPolarizedCopy=*(rhoOutValuesSpinPolarized);

    }

    std::map<dealii::CellId, std::vector<double> > gradRhoOutValuesSpinPolarizedCopy;
    if(dftParameters::spinPolarized==1 && dftParameters::xc_id==4)
    {
      gradRhoOutValuesSpinPolarizedCopy=*(gradRhoOutValuesSpinPolarized);

    }

    //cleanup of existing rho Out and rho In data
    clearRhoData();

    ///copy back temporary rho out to rho in data
    rhoInVals.push_back(rhoOutValuesCopy);
    rhoInValues=&(rhoInVals.back());

    if (dftParameters::xc_id==4)
    {
      gradRhoInVals.push_back(gradRhoOutValuesCopy);
      gradRhoInValues=&(gradRhoInVals.back());
    }

    if(dftParameters::spinPolarized==1)
    {
      rhoInValsSpinPolarized.push_back(rhoOutValuesSpinPolarizedCopy);
      rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
    }

    if (dftParameters::xc_id==4 && dftParameters::spinPolarized==1)
    {
      gradRhoInValsSpinPolarized.push_back(gradRhoOutValuesSpinPolarizedCopy);
      gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
    }

    if(dftParameters::mixingMethod == "ANDERSON_WITH_KERKER")
    {

      d_rhoInNodalValues = d_rhoOutNodalValues;

      //normalize rho
      const double charge = totalCharge(d_matrixFreeDataPRefined,
          d_rhoInNodalValues);

      const double scalingFactor = ((double)numElectrons)/charge;

      //scale nodal vector with scalingFactor
      d_rhoInNodalValues *= scalingFactor;
      d_rhoInNodalVals.push_back(d_rhoInNodalValues);

      interpolateNodalDataToQuadratureData(d_matrixFreeDataPRefined,
          d_rhoInNodalValues,
          *rhoInValues,
          *gradRhoInValues,
          *gradRhoInValues,
          dftParameters::xc_id == 4);


      rhoOutVals.push_back(std::map<dealii::CellId,std::vector<double> > ());
      rhoOutValues = &(rhoOutVals.back());

      if(dftParameters::xc_id == 4)
      {
        gradRhoOutVals.push_back(std::map<dealii::CellId, std::vector<double> >());
        gradRhoOutValues= &(gradRhoOutVals.back());
      }


    }

    //scale quadrature values
    normalizeRho();
  }
}

  template <unsigned int FEOrder>
void dftClass<FEOrder>::computeRhoNodalFromPSI(
#ifdef DFTFE_WITH_GPU
    kohnShamDFTOperatorCUDAClass<FEOrder> & kohnShamDFTEigenOperator,
#endif
    bool isConsiderSpectrumSplitting)
{
  std::map<dealii::CellId, std::vector<double> >  rhoPRefinedNodalData;

  //initialize variables to be used later
  const unsigned int dofs_per_cell = d_dofHandlerPRefined.get_fe().dofs_per_cell;
  typename DoFHandler<3>::active_cell_iterator cell = d_dofHandlerPRefined.begin_active(), endc = d_dofHandlerPRefined.end();
  dealii::IndexSet locallyOwnedDofs = d_dofHandlerPRefined.locally_owned_dofs();
  QGaussLobatto<3>  quadrature_formula(C_num1DKerkerPoly<FEOrder>()+1);
  const unsigned int numQuadPoints = quadrature_formula.size();

  AssertThrow(numQuadPoints == matrix_free_data.get_n_q_points(3),ExcMessage("Number of quadrature points from Quadrature object does not match with number of quadrature points obtained from matrix_free object"));

  //get access to quadrature point coordinates and 2p DoFHandler nodal points
  const std::vector<Point<3> > & quadraturePointCoor = quadrature_formula.get_points();
  const std::vector<Point<3> > & supportPointNaturalCoor = d_dofHandlerPRefined.get_fe().get_unit_support_points();
  std::vector<unsigned int> renumberingMap(numQuadPoints);

  //create renumbering map between the numbering order of quadrature points and lobatto support points
  for(unsigned int i = 0; i < numQuadPoints; ++i)
  {
    const Point<3> & nodalCoor = supportPointNaturalCoor[i];
    for(unsigned int j = 0; j < numQuadPoints; ++j)
    {
      const Point<3> & quadCoor = quadraturePointCoor[j];
      double dist = quadCoor.distance(nodalCoor);
      if(dist <= 1e-08)
      {
        renumberingMap[i] = j;
        break;
      }
    }
  }

  //allocate the storage to compute 2p nodal values from wavefunctions
  for(; cell!=endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
      const dealii::CellId cellId=cell->id();
      rhoPRefinedNodalData[cellId] = std::vector<double>(numQuadPoints,0.0);
    }
  }

  //allocate dummy datastructures
  std::map<dealii::CellId, std::vector<double> > _gradRhoValues;
  std::map<dealii::CellId, std::vector<double> > _rhoValuesSpinPolarized;
  std::map<dealii::CellId, std::vector<double> > _gradRhoValuesSpinPolarized;

  cell = dofHandler.begin_active();
  endc = dofHandler.end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      const dealii::CellId cellId=cell->id();
      if(dftParameters::xc_id == 4)
        (_gradRhoValues)[cellId] = std::vector<double>(3*numQuadPoints,0.0);

      if (dftParameters::spinPolarized==1)
      {
        (_rhoValuesSpinPolarized)[cellId] = std::vector<double>(2*numQuadPoints,0.0);
        if(dftParameters::xc_id == 4)
          (_gradRhoValuesSpinPolarized)[cellId]
            = std::vector<double>(6*numQuadPoints,0.0);
      }
    }


  DensityCalculator<FEOrder> densityCalculator;  

  //compute rho from wavefunctions at nodal locations of 2p DoFHandler nodes in each cell
#ifdef DFTFE_WITH_GPU
  if (dftParameters::useGPU)
    CUDA::computeRhoFromPSI(
        d_eigenVectorsFlattenedCUDA.begin(),
        d_eigenVectorsRotFracFlattenedCUDA.begin(),
        d_numEigenValues,
        d_numEigenValuesRR,
        d_eigenVectorsFlattenedSTL[0].size()/d_numEigenValues,
        eigenValues,
        fermiEnergy,
        fermiEnergyUp,
        fermiEnergyDown,
        kohnShamDFTEigenOperator,
        dofHandler,
        matrix_free_data.n_physical_cells(),
        matrix_free_data.get_dofs_per_cell(),
        quadrature_formula.size(),
        d_kPointWeights,
        &rhoPRefinedNodalData,
        &_gradRhoValues,
        &_rhoValuesSpinPolarized,
        &_gradRhoValuesSpinPolarized,
        false,
        interpoolcomm,
        interBandGroupComm,
        isConsiderSpectrumSplitting && d_numEigenValues!=d_numEigenValuesRR,
        true);

  else
    densityCalculator.computeRhoFromPSI(
        d_eigenVectorsFlattenedSTL,
        d_eigenVectorsRotFracDensityFlattenedSTL,
        d_numEigenValues,
        d_numEigenValuesRR,
        eigenValues,
        fermiEnergy,
        fermiEnergyUp,
        fermiEnergyDown,
        dofHandler,
        constraintsNone,
        matrix_free_data,
        eigenDofHandlerIndex,
        3,
        localProc_dof_indicesReal,
        localProc_dof_indicesImag,        
        d_kPointWeights,
        &rhoPRefinedNodalData,
        &_gradRhoValues,
        &_rhoValuesSpinPolarized,
        &_gradRhoValuesSpinPolarized,
        dftParameters::xc_id == 4,
        interpoolcomm,
        interBandGroupComm,
        isConsiderSpectrumSplitting,
        true);
#else
  densityCalculator.computeRhoFromPSI(
      d_eigenVectorsFlattenedSTL,
      d_eigenVectorsRotFracDensityFlattenedSTL,
      d_numEigenValues,
      d_numEigenValuesRR,
      eigenValues,
      fermiEnergy,
      fermiEnergyUp,
      fermiEnergyDown,
      dofHandler,
      constraintsNone,
      matrix_free_data,
      eigenDofHandlerIndex,
      3,
      localProc_dof_indicesReal,
      localProc_dof_indicesImag,
      d_kPointWeights,
      &rhoPRefinedNodalData,
      &_gradRhoValues,
      &_rhoValuesSpinPolarized,
      &_gradRhoValuesSpinPolarized,
      dftParameters::xc_id == 4,
      interpoolcomm,
      interBandGroupComm,
      isConsiderSpectrumSplitting,
      true);
#endif

  //copy Lobatto quadrature data to fill in 2p DoFHandler nodal data  
  DoFHandler<3>::active_cell_iterator
    cellP = d_dofHandlerPRefined.begin_active(),
          endcP = d_dofHandlerPRefined.end();

  for (; cellP!=endcP; ++cellP)
  {
    if (cellP->is_locally_owned())
    {
      std::vector<dealii::types::global_dof_index> cell_dof_indices(dofs_per_cell);
      cellP->get_dof_indices(cell_dof_indices);
      const std::vector<double> & nodalValues = rhoPRefinedNodalData.find(cellP->id())->second;
      Assert(nodalValues.size() == dofs_per_cell,ExcMessage("Number of nodes in 2p DoFHandler does not match with data stored in rhoNodal Values variable"));

      for(unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
      {
        const dealii::types::global_dof_index nodeID = cell_dof_indices[iNode];
        if(!d_constraintsPRefined.is_constrained(nodeID))
        {
          if(locallyOwnedDofs.is_element(nodeID))
            d_rhoOutNodalValues(nodeID) =  nodalValues[renumberingMap[iNode]];
        }
      }
    }
  }


}
