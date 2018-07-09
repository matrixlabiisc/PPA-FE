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


template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeNonLocalHamiltonianTimesX(const std::vector<vectorType> &src,
							   std::vector<vectorType>       &dst)const
{

  //
  //get access to triangulation objects from meshGenerator class
  //
  const unsigned int kPointIndex = d_kPointIndex;
  const unsigned int dofs_per_cell = dftPtr->FEEigen.dofs_per_cell;


#ifdef USE_COMPLEX
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;//GeometryInfo<3>::vertices_per_cell;
#else
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  //
  //compute nonlocal projector ket times x i.e C^{T}*X
  //
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
#ifdef USE_COMPLEX
  std::map<unsigned int, std::vector<std::complex<double> > > projectorKetTimesVector;
#else
  std::map<unsigned int, std::vector<double> > projectorKetTimesVector;
#endif

  const unsigned int numberWaveFunctions = src.size();
  projectorKetTimesVector.clear();

  //
  //allocate memory for matrix-vector product
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[atomId].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }

  //
  //some useful vectors
  //
#ifdef USE_COMPLEX
  std::vector<std::complex<double> > inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
  std::vector<double> inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif


  //
  //parallel loop over all elements to compute nonlocal projector ket times x i.e C^{T}*X
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandlerEigen.begin_active(), endc = dftPtr->dofHandlerEigen.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem ++;
	  cell->get_dof_indices(local_dof_indices);

	  unsigned int index=0;

	  std::vector<double> temp(dofs_per_cell,0.0);
	  if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size()>0)
	  {
	    for (std::vector<vectorType>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
#ifdef USE_COMPLEX
	      (*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), temp.begin());
	      for(unsigned int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  //
		  //This is the component index 0(real) or 1(imag).
		  //
		  const unsigned int ck = dftPtr->FEEigen.system_to_component_index(idof).first;
		  const unsigned int iNode = dftPtr->FEEigen.system_to_component_index(idof).second;
		  if(ck == 0)
		    inputVectors[numberNodesPerElement*index + iNode].real(temp[idof]);
		  else
		    inputVectors[numberNodesPerElement*index + iNode].imag(temp[idof]);
		}
#else
	      (*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), inputVectors.begin()+numberNodesPerElement*index);
#endif
	      index++;
	    }
	  }


	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];
#ifdef USE_COMPLEX
	      const char transA = 'C';
	      const char transB = 'N';
	      const std::complex<double> alpha = 1.0;
	      const std::complex<double> beta = 1.0;
	      zgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberPseudoWaveFunctions);
#else
	      const char transA = 'T';
	      const char transB = 'N';
	      const double alpha = 1.0;
	      const double beta = 1.0;
	      dgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberPseudoWaveFunctions);
#endif
	    }

	}

    }//element loop

#ifdef USE_COMPLEX
  dftPtr->d_projectorKetTimesVectorParFlattened=std::complex<double>(0.0,0.0);
#else
  dftPtr->d_projectorKetTimesVectorParFlattened=0.0;
#endif


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	      dftPtr->d_projectorKetTimesVectorParFlattened[dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]*numberWaveFunctions+iWave]=projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave];
    }

  dftPtr->d_projectorKetTimesVectorParFlattened.compress(VectorOperation::add);
  dftPtr->d_projectorKetTimesVectorParFlattened.update_ghost_values();


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {

              projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave] =dftPtr->d_projectorKetTimesVectorParFlattened[dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]*numberWaveFunctions+iWave];

	    }
	}
    }


  //
  //compute V*C^{T}*X
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    projectorKetTimesVector[atomId][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
	}
    }

  //std::cout<<"Scaling V*C^{T} "<<std::endl;

  const char transA1 = 'N';
  const char transB1 = 'N';

  //
  //access elementIdsInAtomCompactSupport
  //

#ifdef USE_COMPLEX
  std::vector<std::complex<double> > outputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
  std::vector<double> outputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif

  //
  //compute C*V*C^{T}*x
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size(); ++iElemComp)
	{

	  DoFHandler<3>::active_cell_iterator cell = dftPtr->d_elementIteratorsInAtomCompactSupport[atomId][iElemComp];

#ifdef USE_COMPLEX
	  const std::complex<double> alpha1 = 1.0;
	  const std::complex<double> beta1 = 0.0;

	  zgemm_(&transA1,
		 &transB1,
		 &numberNodesPerElement,
		 &numberWaveFunctions,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &dftPtr->d_nonLocalProjectorElementMatrices[atomId][iElemComp][kPointIndex][0],
		 &numberNodesPerElement,
		 &projectorKetTimesVector[atomId][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &outputVectors[0],
		 &numberNodesPerElement);
#else
	  const double alpha1 = 1.0;
	  const double beta1 = 0.0;

	  dgemm_(&transA1,
		 &transB1,
		 &numberNodesPerElement,
		 &numberWaveFunctions,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &dftPtr->d_nonLocalProjectorElementMatrices[atomId][iElemComp][kPointIndex][0],
		 &numberNodesPerElement,
		 &projectorKetTimesVector[atomId][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &outputVectors[0],
		 &numberNodesPerElement);
#endif

	  cell->get_dof_indices(local_dof_indices);

#ifdef USE_COMPLEX
	  unsigned int index = 0;
	  std::vector<double> temp(dofs_per_cell,0.0);
	  for(std::vector<vectorType>::iterator it = dst.begin(); it != dst.end(); ++it)
	    {
	      for(unsigned int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  const unsigned int ck = dftPtr->FEEigen.system_to_component_index(idof).first;
		  const unsigned int iNode = dftPtr->FEEigen.system_to_component_index(idof).second;

		  if(ck == 0)
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].real();
		  else
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].imag();

		}
	      dftPtr->constraintsNoneEigen.distribute_local_to_global(temp.begin(), temp.end(),local_dof_indices.begin(), *it);
	      index++;
	    }
#else
	  std::vector<double>::iterator iter = outputVectors.begin();
	  for (std::vector<vectorType>::iterator it=dst.begin(); it!=dst.end(); ++it)
	    {
	      dftPtr->constraintsNoneEigen.distribute_local_to_global(iter, iter+numberNodesPerElement,local_dof_indices.begin(), *it);
	      iter+=numberNodesPerElement;
	    }
#endif

	}

    }
}


#ifdef USE_COMPLEX
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeNonLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<std::complex<double> > & src,
							   const unsigned int numberWaveFunctions,
							   dealii::parallel::distributed::Vector<std::complex<double> >       & dst) const
{

  std::map<unsigned int, std::vector<std::complex<double> > > projectorKetTimesVector;

  //
  //allocate memory for matrix-vector product
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[atomId].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }


  std::vector<std::complex<double> > cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //
  //blas required settings
  //
  const char transA = 'N';
  const char transB = 'N';
  const std::complex<double> alpha = 1.0;
  const std::complex<double> beta = 1.0;
  const unsigned int inc = 1;


  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem++;
	  if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size()>0)
	  {
	    for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      zcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
	    }
	  }

	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

	      zgemm_(&transA,
		     &transB,
		     &numberWaveFunctions,
		     &numberPseudoWaveFunctions,
		     &d_numberNodesPerElement,
		     &alpha,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &dftPtr->d_nonLocalProjectorElementMatricesConjugate[atomId][nonZeroElementMatrixId][d_kPointIndex][0],
		     &d_numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberWaveFunctions);
	    }


	}

    }//cell loop

  dftPtr->d_projectorKetTimesVectorParFlattened=std::complex<double>(0.0,0.0);


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
      {
        const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];
	zcopy_(&numberWaveFunctions,
                &projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave],
                &inc,
                &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
                &inc);
      }
    }


  dftPtr->d_projectorKetTimesVectorParFlattened.compress(VectorOperation::add);
  dftPtr->d_projectorKetTimesVectorParFlattened.update_ghost_values();

  //
  //compute V*C^{T}*X
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
      {
	  std::complex<double> nonlocalConstantV;
	  nonlocalConstantV.real(dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave]);
	  nonlocalConstantV.imag(0);

          const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];

	  zscal_(&numberWaveFunctions,
		 &nonlocalConstantV,
		 &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
		 &inc);

	  zcopy_(&numberWaveFunctions,
		 &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
		 &inc,
		 &projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave],
		 &inc);
      }

    }


  std::vector<std::complex<double> > cellNonLocalHamTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //blas required settings
  const char transA1 = 'N';
  const char transB1 = 'N';
  const std::complex<double> alpha1 = 1.0;
  const std::complex<double> beta1 = 0.0;
  const unsigned int inc1 = 1;

  //
  //compute C*V*C^{T}*x
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
       for(unsigned int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size(); ++iElemComp)
	{

	  zgemm_(&transA1,
	         &transB1,
                 &numberWaveFunctions,
                 &d_numberNodesPerElement,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &projectorKetTimesVector[atomId][0],
		 &numberWaveFunctions,
		 &dftPtr->d_nonLocalProjectorElementMatricesTranspose[atomId][iElemComp][d_kPointIndex][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &cellNonLocalHamTimesWaveMatrix[0],
		 &numberWaveFunctions);

        unsigned int elementId =  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp];

	for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[elementId][iNode];
	      zaxpy_(&numberWaveFunctions,
		     &alpha1,
		     &cellNonLocalHamTimesWaveMatrix[numberWaveFunctions*iNode],
		     &inc1,
		     dst.begin()+localNodeId,
		     &inc1);
	    }


	}

    }

}
#else
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeNonLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<double> & src,
							   const unsigned int numberWaveFunctions,
							   dealii::parallel::distributed::Vector<double>       & dst) const
{


  std::map<unsigned int, std::vector<double> > projectorKetTimesVector;

  //
  //allocate memory for matrix-vector product
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[atomId].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }


  std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //
  //blas required settings
  //
  const char transA = 'N';
  const char transB = 'N';
  const double alpha = 1.0;
  const double beta = 1.0;
  const unsigned int inc = 1;


  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem++;
	  if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size()>0)
	  {
	    for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      dcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
	    }
	  }

	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

	      dgemm_(&transA,
		     &transB,
		     &numberWaveFunctions,
		     &numberPseudoWaveFunctions,
		     &d_numberNodesPerElement,
		     &alpha,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][d_kPointIndex][0],
		     &d_numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberWaveFunctions);
	    }


	}

    }//cell loop

  dftPtr->d_projectorKetTimesVectorParFlattened=0.0;

  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
      {
          const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];

          dcopy_(&numberWaveFunctions,
                &projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave],
                &inc,
                &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
                &inc);

      }
    }

  dftPtr->d_projectorKetTimesVectorParFlattened.compress(VectorOperation::add);
  dftPtr->d_projectorKetTimesVectorParFlattened.update_ghost_values();

  //
  //compute V*C^{T}*X
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	{
	  double nonlocalConstantV=dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];

          const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];

	  dscal_(&numberWaveFunctions,
		 &nonlocalConstantV,
		 &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
		 &inc);

	  dcopy_(&numberWaveFunctions,
		 &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
		 &inc,
		 &projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave],
		 &inc);

	}

    }


  std::vector<double> cellNonLocalHamTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //blas required settings
  const char transA1 = 'N';
  const char transB1 = 'N';
  const double alpha1 = 1.0;
  const double beta1 = 0.0;
  const unsigned int inc1 = 1;

  //
  //compute C*V*C^{T}*x
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
       for(unsigned int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size(); ++iElemComp)
	{

	  dgemm_(&transA1,
	         &transB1,
                 &numberWaveFunctions,
                 &d_numberNodesPerElement,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &projectorKetTimesVector[atomId][0],
		 &numberWaveFunctions,
		 &dftPtr->d_nonLocalProjectorElementMatricesTranspose[atomId][iElemComp][d_kPointIndex][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &cellNonLocalHamTimesWaveMatrix[0],
		 &numberWaveFunctions);

        unsigned int elementId =  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp];

	for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[elementId][iNode];
	      daxpy_(&numberWaveFunctions,
		     &alpha1,
		     &cellNonLocalHamTimesWaveMatrix[numberWaveFunctions*iNode],
		     &inc1,
		     dst.begin()+localNodeId,
		     &inc1);
	    }


	}

    }

}
#endif
