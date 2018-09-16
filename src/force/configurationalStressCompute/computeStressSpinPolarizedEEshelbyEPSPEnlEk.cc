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
// @author Sambit Das
//
#ifdef USE_COMPLEX
//compute configurational stress contribution from all terms except the nuclear self energy
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStressSpinPolarizedEEshelbyEPSPEnlEk()
{
  std::vector<std::vector<vectorType>> eigenVectors((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());
  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size(); ++kPoint)
  {
        eigenVectors[kPoint].resize(dftPtr->numEigenValues);
        for(unsigned int i = 0; i < dftPtr->numEigenValues; ++i)
          eigenVectors[kPoint][i].reinit(dftPtr->d_tempEigenVec);

#ifdef USE_COMPLEX
	vectorTools::copyFlattenedDealiiVecToSingleCompVec
		 (dftPtr->d_eigenVectorsFlattened[kPoint],
		  dftPtr->numEigenValues,
		  std::make_pair(0,dftPtr->numEigenValues),
		  dftPtr->localProc_dof_indicesReal,
		  dftPtr->localProc_dof_indicesImag,
		  eigenVectors[kPoint]);
#else
	vectorTools::copyFlattenedDealiiVecToSingleCompVec
		 (dftPtr->d_eigenVectorsFlattened[kPoint],
		  dftPtr->numEigenValues,
		  std::make_pair(0,dftPtr->numEigenValues),
		  eigenVectors[kPoint]);
#endif
  }

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIds.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const bool isPseudopotential = dftParameters::isPseudopotential;
  const unsigned int numVectorizedArrayElements=VectorizedArray<double>::n_array_elements;
  const MatrixFree<3,double> & matrix_free_data=dftPtr->matrix_free_data;

  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(matrix_free_data,
	                                                        d_forceDofHandlerIndex,
								0);
  FEEvaluation<C_DIM,1,C_num1DQuadPSP<FEOrder>(),C_DIM>  forceEvalNLP(matrix_free_data,
	                                                           d_forceDofHandlerIndex,
								   2);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),2> psiEvalSpin0(matrix_free_data,
	                                                            dftPtr->eigenDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),2> psiEvalSpin1(matrix_free_data,
	                                                            dftPtr->eigenDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),2> psiEvalSpin0NLP(matrix_free_data,
	                                                            dftPtr->eigenDofHandlerIndex,
								    2);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),2> psiEvalSpin1NLP(matrix_free_data,
	                                                            dftPtr->eigenDofHandlerIndex,
								    2);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEval(matrix_free_data,
	                                                          dftPtr->phiTotDofHandlerIndex,
								  0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotInEval(matrix_free_data,
	                                                            dftPtr->phiTotDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiExtEval(matrix_free_data,
	                                                          dftPtr->phiExtDofHandlerIndex,
								  0);
  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<C_DIM> feVselfValues (dftPtr->FE, quadrature, update_gradients | update_quadrature_points);

  const unsigned int numQuadPoints=forceEval.n_q_points;
  const unsigned int numQuadPointsNLP=dftParameters::useHigherQuadNLP?
                                      forceEvalNLP.n_q_points:numQuadPoints;
  const unsigned int numEigenVectors=dftPtr->numEigenValues;
  const unsigned int numKPoints=dftPtr->d_kPointWeights.size();

  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
  Tensor<1,2,VectorizedArray<double> > zeroTensor1;zeroTensor1[0]=make_vectorized_array(0.0);zeroTensor1[1]=make_vectorized_array(0.0);
  Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > zeroTensor2;
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor3;
  Tensor<2,C_DIM,VectorizedArray<double> > zeroTensor4;
  Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > zeroTensor5;
  for (unsigned int idim=0; idim<C_DIM; idim++){
    zeroTensor2[0][idim]=make_vectorized_array(0.0);
    zeroTensor2[1][idim]=make_vectorized_array(0.0);
    zeroTensor3[idim]=make_vectorized_array(0.0);
  }
  for (unsigned int idim=0; idim<C_DIM; idim++)
  {
    for (unsigned int jdim=0; jdim<C_DIM; jdim++)
    {
	zeroTensor4[idim][jdim]=make_vectorized_array(0.0);
    }
  }
  zeroTensor5[0]=zeroTensor4;
  zeroTensor5[1]=zeroTensor4;

  VectorizedArray<double> phiExtFactor=make_vectorized_array(0.0);
  std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin0TimesV(numKPoints);
  std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin1TimesV(numKPoints);
  if (isPseudopotential)
  {
    phiExtFactor=make_vectorized_array(1.0);
    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
    {
         computeNonLocalProjectorKetTimesPsiTimesVFlattened(dftPtr->d_eigenVectorsFlattened[2*ikPoint],
		                                   numEigenVectors,
                                                   projectorKetTimesPsiSpin0TimesV[ikPoint],
						   ikPoint);
    }
    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
    {
         computeNonLocalProjectorKetTimesPsiTimesVFlattened(dftPtr->d_eigenVectorsFlattened[2*ikPoint+1],
		                                   numEigenVectors,
                                                   projectorKetTimesPsiSpin1TimesV[ikPoint],
						   ikPoint);
    }
  }

  std::vector<VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin0Quads(numQuadPoints,zeroTensor3);
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin1Quads(numQuadPoints,zeroTensor3);
  std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > pseudoVLocQuads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPseudoVLocQuads(numQuadPoints,zeroTensor3);
  std::vector<VectorizedArray<double> > vEffRhoInSpin0Quads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > vEffRhoInSpin1Quads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > vEffRhoOutSpin0Quads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > vEffRhoOutSpin1Quads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoInSpin0Quads(numQuadPoints,zeroTensor3);
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoInSpin1Quads(numQuadPoints,zeroTensor3);
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints,zeroTensor3);
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints,zeroTensor3);

  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell)
  {
    forceEval.reinit(cell);
    psiEvalSpin0.reinit(cell);
    psiEvalSpin1.reinit(cell);

    if (isPseudopotential && dftParameters::useHigherQuadNLP)
    {
      forceEvalNLP.reinit(cell);
      psiEvalSpin0NLP.reinit(cell);
      psiEvalSpin1NLP.reinit(cell);
    }

    phiTotEval.reinit(cell);
    phiTotEval.read_dof_values_plain(dftPtr->d_phiTotRhoOut);//read without taking constraints into account
    phiTotEval.evaluate(true,true);

    phiTotInEval.reinit(cell);
    phiTotInEval.read_dof_values_plain(dftPtr->d_phiTotRhoIn);//read without taking constraints into account
    phiTotInEval.evaluate(true,true);

    phiExtEval.reinit(cell);
    phiExtEval.read_dof_values_plain(dftPtr->d_phiExt);
    phiExtEval.evaluate(true,true);

    std::fill(rhoQuads.begin(),rhoQuads.end(),make_vectorized_array(0.0));
    std::fill(gradRhoSpin0Quads.begin(),gradRhoSpin0Quads.end(),zeroTensor3);
    std::fill(gradRhoSpin1Quads.begin(),gradRhoSpin1Quads.end(),zeroTensor3);
    std::fill(excQuads.begin(),excQuads.end(),make_vectorized_array(0.0));
    std::fill(pseudoVLocQuads.begin(),pseudoVLocQuads.end(),make_vectorized_array(0.0));
    std::fill(gradPseudoVLocQuads.begin(),gradPseudoVLocQuads.end(),zeroTensor3);
    std::fill(vEffRhoInSpin0Quads.begin(),vEffRhoInSpin0Quads.end(),make_vectorized_array(0.0));
    std::fill(vEffRhoInSpin1Quads.begin(),vEffRhoInSpin1Quads.end(),make_vectorized_array(0.0));
    std::fill(vEffRhoOutSpin0Quads.begin(),vEffRhoOutSpin0Quads.end(),make_vectorized_array(0.0));
    std::fill(vEffRhoOutSpin1Quads.begin(),vEffRhoOutSpin1Quads.end(),make_vectorized_array(0.0));
    std::fill(derExchCorrEnergyWithGradRhoInSpin0Quads.begin(),derExchCorrEnergyWithGradRhoInSpin0Quads.end(),zeroTensor3);
    std::fill(derExchCorrEnergyWithGradRhoInSpin1Quads.begin(),derExchCorrEnergyWithGradRhoInSpin1Quads.end(),zeroTensor3);
    std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),zeroTensor3);
    std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),zeroTensor3);
    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
	 vEffRhoInSpin0Quads[q]=phiTotInEval.get_value(q);
	 vEffRhoInSpin1Quads[q]=phiTotInEval.get_value(q);
	 vEffRhoOutSpin0Quads[q]=phiTotEval.get_value(q);
	 vEffRhoOutSpin1Quads[q]=phiTotEval.get_value(q);
    }
    //allocate storage for vector of quadPoints, nonlocal atom id, pseudo wave, k point
    //FIXME: flatten nonlocal atomid id and pseudo wave and k point
    std::vector<std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > >ZetaDeltaVQuads;
    std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > > > > >gradZetalmDeltaVlDyadicDistImageAtomsQuads;
    if(isPseudopotential)
    {
	ZetaDeltaVQuads.resize(numQuadPointsNLP);
	gradZetalmDeltaVlDyadicDistImageAtomsQuads.resize(numQuadPointsNLP);
	for (unsigned int q=0; q<numQuadPointsNLP; ++q)
	{
	  ZetaDeltaVQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
	  gradZetalmDeltaVlDyadicDistImageAtomsQuads[q].resize(d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.size());
	  for (unsigned int i=0; i < d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.size(); ++i)
	  {
	    const int numberPseudoWaveFunctions = d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i].size();
	    ZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions);
	    gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i].resize(numberPseudoWaveFunctions);
	    for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	    {
	        ZetaDeltaVQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor1);
		gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor5);
	    }
	  }
	}
    }

    const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);
    //For LDA
    std::vector<double> exchValRhoOut(numQuadPoints);
    std::vector<double> corrValRhoOut(numQuadPoints);
    std::vector<double> exchPotValRhoOut(2*numQuadPoints);
    std::vector<double> corrPotValRhoOut(2*numQuadPoints);
    std::vector<double> exchValRhoIn(numQuadPoints);
    std::vector<double> corrValRhoIn(numQuadPoints);
    std::vector<double> exchPotValRhoIn(2*numQuadPoints);
    std::vector<double> corrPotValRhoIn(2*numQuadPoints);
    //
    //For GGA
    std::vector<double> sigmaValRhoOut(3*numQuadPoints);
    std::vector<double> derExchEnergyWithDensityValRhoOut(2*numQuadPoints), derCorrEnergyWithDensityValRhoOut(2*numQuadPoints), derExchEnergyWithSigmaRhoOut(3*numQuadPoints),derCorrEnergyWithSigmaRhoOut(3*numQuadPoints);
    std::vector<double> sigmaValRhoIn(3*numQuadPoints);
    std::vector<double> derExchEnergyWithDensityValRhoIn(2*numQuadPoints), derCorrEnergyWithDensityValRhoIn(2*numQuadPoints), derExchEnergyWithSigmaRhoIn(3*numQuadPoints),derCorrEnergyWithSigmaRhoIn(3*numQuadPoints);
    std::vector<Tensor<1,C_DIM,double > > gradRhoInSpin0(numQuadPoints);
    std::vector<Tensor<1,C_DIM,double > > gradRhoInSpin1(numQuadPoints);
    std::vector<Tensor<1,C_DIM,double > > gradRhoOutSpin0(numQuadPoints);
    std::vector<Tensor<1,C_DIM,double > > gradRhoOutSpin1(numQuadPoints);
    //
    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
    {
       subCellPtr= matrix_free_data.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId=subCellPtr->id();
       if(dftParameters::xc_id == 4)
       {
	  for (unsigned int q = 0; q < numQuadPoints; ++q)
	  {
	      for (unsigned int idim=0; idim<C_DIM; idim++)
	      {
	        gradRhoOutSpin0[q][idim] = ((*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q + idim]);
	        gradRhoOutSpin1[q][idim] = ((*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q +3+idim]);
	      }
	      sigmaValRhoOut[3*q+0] = scalar_product(gradRhoOutSpin0[q],gradRhoOutSpin0[q]);
	      sigmaValRhoOut[3*q+1] = scalar_product(gradRhoOutSpin0[q],gradRhoOutSpin1[q]);
	      sigmaValRhoOut[3*q+2] = scalar_product(gradRhoOutSpin1[q],gradRhoOutSpin1[q]);

	      for (unsigned int idim=0; idim<C_DIM; idim++)
	      {
	        gradRhoInSpin0[q][idim] = ((*dftPtr->gradRhoInValuesSpinPolarized)[subCellId][6*q + idim]);
	        gradRhoInSpin1[q][idim] = ((*dftPtr->gradRhoInValuesSpinPolarized)[subCellId][6*q +3+idim]);
	      }
	      sigmaValRhoIn[3*q+0] = scalar_product(gradRhoInSpin0[q],gradRhoInSpin0[q]);
	      sigmaValRhoIn[3*q+1] = scalar_product(gradRhoInSpin0[q],gradRhoInSpin1[q]);
	      sigmaValRhoIn[3*q+2] = scalar_product(gradRhoInSpin1[q],gradRhoInSpin1[q]);
	  }
	  xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&sigmaValRhoOut[0],&exchValRhoOut[0],&derExchEnergyWithDensityValRhoOut[0],&derExchEnergyWithSigmaRhoOut[0]);
	  xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&sigmaValRhoOut[0],&corrValRhoOut[0],&derCorrEnergyWithDensityValRhoOut[0],&derCorrEnergyWithSigmaRhoOut[0]);
	  xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&sigmaValRhoIn[0],&exchValRhoIn[0],&derExchEnergyWithDensityValRhoIn[0],&derExchEnergyWithSigmaRhoIn[0]);
	  xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&sigmaValRhoIn[0],&corrValRhoIn[0],&derCorrEnergyWithDensityValRhoIn[0],&derCorrEnergyWithSigmaRhoIn[0]);
          for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
	     vEffRhoInSpin0Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoIn[2*q]+derCorrEnergyWithDensityValRhoIn[2*q];
	     vEffRhoInSpin1Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoIn[2*q+1]+derCorrEnergyWithDensityValRhoIn[2*q+1];
             vEffRhoOutSpin0Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoOut[2*q]+derCorrEnergyWithDensityValRhoOut[2*q];
             vEffRhoOutSpin1Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoOut[2*q+1]+derCorrEnergyWithDensityValRhoOut[2*q+1];
	      for (unsigned int idim=0; idim<C_DIM; idim++)
	      {
	         derExchCorrEnergyWithGradRhoInSpin0Quads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoIn[3*q+0]+derCorrEnergyWithSigmaRhoIn[3*q+0])*gradRhoInSpin0[q][idim];
	         derExchCorrEnergyWithGradRhoInSpin0Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoIn[3*q+1]+derCorrEnergyWithSigmaRhoIn[3*q+1])*gradRhoInSpin1[q][idim];

	         derExchCorrEnergyWithGradRhoInSpin1Quads[q][idim][iSubCell]+=2.0*(derExchEnergyWithSigmaRhoIn[3*q+2]+derCorrEnergyWithSigmaRhoIn[3*q+2])*gradRhoInSpin1[q][idim];
	         derExchCorrEnergyWithGradRhoInSpin1Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoIn[3*q+1]+derCorrEnergyWithSigmaRhoIn[3*q+1])*gradRhoInSpin0[q][idim];

	         derExchCorrEnergyWithGradRhoOutSpin0Quads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoOut[3*q+0]+derCorrEnergyWithSigmaRhoOut[3*q+0])*gradRhoOutSpin0[q][idim];
	         derExchCorrEnergyWithGradRhoOutSpin0Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoOut[3*q+1]+derCorrEnergyWithSigmaRhoOut[3*q+1])*gradRhoOutSpin1[q][idim];

	         derExchCorrEnergyWithGradRhoOutSpin1Quads[q][idim][iSubCell]+=2.0*(derExchEnergyWithSigmaRhoOut[3*q+2]+derCorrEnergyWithSigmaRhoOut[3*q+2])*gradRhoOutSpin1[q][idim];
	         derExchCorrEnergyWithGradRhoOutSpin1Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoOut[3*q+1]+derCorrEnergyWithSigmaRhoOut[3*q+1])*gradRhoOutSpin0[q][idim];
	      }
          }

       }
       else
       {
          xc_lda_exc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&exchValRhoOut[0]);
          xc_lda_exc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&corrValRhoOut[0]);
	  xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&exchPotValRhoOut[0]);
	  xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&corrPotValRhoOut[0]);
	  xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&exchPotValRhoIn[0]);
	  xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&corrPotValRhoIn[0]);
          for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
	     vEffRhoInSpin0Quads[q][iSubCell]+= exchPotValRhoIn[2*q]+corrPotValRhoIn[2*q];
	     vEffRhoInSpin1Quads[q][iSubCell]+= exchPotValRhoIn[2*q+1]+corrPotValRhoIn[2*q+1];
             vEffRhoOutSpin0Quads[q][iSubCell]+= exchPotValRhoOut[2*q]+corrPotValRhoOut[2*q];
             vEffRhoOutSpin1Quads[q][iSubCell]+= exchPotValRhoOut[2*q+1]+corrPotValRhoOut[2*q+1];

          }
       }

       for (unsigned int q=0; q<numQuadPoints; ++q)
       {
         rhoQuads[q][iSubCell]=(*dftPtr->rhoOutValues)[subCellId][q];
       }
    }

    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);
    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);

    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
        for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
        {
          psiEvalSpin0.read_dof_values_plain(eigenVectors[2*ikPoint][iEigenVec]);
          psiEvalSpin0.evaluate(true,true);
          psiEvalSpin1.read_dof_values_plain(eigenVectors[2*ikPoint+1][iEigenVec]);
          psiEvalSpin1.evaluate(true,true);

          for (unsigned int q=0; q<numQuadPoints; ++q)
          {
	     const int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
             psiSpin0Quads[id]=psiEvalSpin0.get_value(q);
	     psiSpin1Quads[id]=psiEvalSpin1.get_value(q);
             gradPsiSpin0Quads[id]=psiEvalSpin0.get_gradient(q);
	     gradPsiSpin1Quads[id]=psiEvalSpin1.get_gradient(q);

             const double partOccSpin0 =dftUtils::getPartialOccupancy
		                                                     (dftPtr->eigenValues[ikPoint][iEigenVec],
		                                                      dftPtr->fermiEnergy,
								      C_kb,
								      dftParameters::TVal);
             const double partOccSpin1 =dftUtils::getPartialOccupancy
		                                                     (dftPtr->eigenValues[ikPoint][iEigenVec+numEigenVectors],
		                                                      dftPtr->fermiEnergy,
								      C_kb,
								      dftParameters::TVal);
	     const VectorizedArray<double> factor0=make_vectorized_array(dftPtr->d_kPointWeights[ikPoint]*partOccSpin0);
	     const VectorizedArray<double> factor1=make_vectorized_array(dftPtr->d_kPointWeights[ikPoint]*partOccSpin1);

	     gradRhoSpin0Quads[q]+=factor0*internalforce::computeGradRhoContribution(psiSpin0Quads[id],gradPsiSpin0Quads[id]);
	     gradRhoSpin1Quads[q]+=factor1*internalforce::computeGradRhoContribution(psiSpin1Quads[id],gradPsiSpin1Quads[id]);

          }//quad point loop
        } //eigenvector loop

    //accumulate grad rho quad point contribution from all pools
    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
      for (unsigned int q=0; q<numQuadPoints; ++q)
	for (unsigned int idim=0; idim<C_DIM; idim++)
	{
	  gradRhoSpin0Quads[q][idim][iSubCell]=Utilities::MPI::sum(gradRhoSpin0Quads[q][idim][iSubCell],dftPtr->interpoolcomm);
	  gradRhoSpin1Quads[q][idim][iSubCell]=Utilities::MPI::sum(gradRhoSpin1Quads[q][idim][iSubCell],dftPtr->interpoolcomm);
	}

    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin0QuadsNLP;
    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin1QuadsNLP;
    if (isPseudopotential && dftParameters::useHigherQuadNLP)
    {
	psiSpin0QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor1);
	psiSpin1QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor1);
	for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
	    for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
	    {
	      psiEvalSpin0NLP.read_dof_values_plain(eigenVectors[2*ikPoint][iEigenVec]);
	      psiEvalSpin0NLP.evaluate(true,false);

	      psiEvalSpin1NLP.read_dof_values_plain(eigenVectors[2*ikPoint+1][iEigenVec]);
	      psiEvalSpin1NLP.evaluate(true,false);

	      for (unsigned int q=0; q<numQuadPointsNLP; ++q)
	      {
		 const int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
		 psiSpin0QuadsNLP[id]=psiEvalSpin0NLP.get_value(q);
		 psiSpin1QuadsNLP[id]=psiEvalSpin1NLP.get_value(q);
	      }//quad point loop
	    } //eigenvector loop
    }

    if(isPseudopotential)
    {
       for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
       {
          subCellPtr= matrix_free_data.get_cell_iterator(cell,iSubCell);
          dealii::CellId subCellId=subCellPtr->id();
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     pseudoVLocQuads[q][iSubCell]=dftPtr->pseudoValues[subCellId][q];
	     gradPseudoVLocQuads[q][0][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+0];
             gradPseudoVLocQuads[q][1][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+1];
	     gradPseudoVLocQuads[q][2][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+2];
	  }

	  for (unsigned int q=0; q<numQuadPointsNLP; ++q)
	  {
            for (unsigned int i=0; i < d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.size(); ++i)
	    {
	      const int numberPseudoWaveFunctions = d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i].size();
	      for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	      {
		if (d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave].find(subCellId)!=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave].end())
		{
                   for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		   {
                      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][0][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+0];
                      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][1][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+1];
		      for (unsigned int idim=0; idim<C_DIM; idim++)
		      {
		        for (unsigned int jdim=0; jdim<C_DIM; jdim++)
		        {
                           gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][0][idim][jdim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*C_DIM*2+q*C_DIM*C_DIM*2+idim*C_DIM*2+jdim*2+0];
                           gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][1][idim][jdim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*C_DIM*2+q*C_DIM*C_DIM*2+idim*C_DIM*2+jdim*2+1];
			}
		      }
		   }
		}//non-trivial cellId check
	      }//iPseudoWave loop
	    }//i loop
	  }//q loop
       }//subcell loop

       addEPSPStressContribution(feVselfValues,
	                         forceEval,
			         cell,
			         rhoQuads);
    }//is pseudopotential check

    Tensor<2,C_DIM,VectorizedArray<double> > EQuadSum=zeroTensor4;
    Tensor<2,C_DIM,VectorizedArray<double> > EKPointsQuadSum=zeroTensor4;
    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
       VectorizedArray<double> phiTot_q =phiTotEval.get_value(q);
       Tensor<1,C_DIM,VectorizedArray<double> > gradPhiTot_q =phiTotEval.get_gradient(q);
       VectorizedArray<double> phiExt_q =phiExtEval.get_value(q)*phiExtFactor;

       Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getEElectroEshelbyTensor
	                                                     (phiTot_q,
			                                      gradPhiTot_q,
						              rhoQuads[q],
							      pseudoVLocQuads[q],
							      phiExt_q);

       E+=eshelbyTensorSP::getELocXcPspEshelbyTensor
				      (rhoQuads[q],
				       gradRhoSpin0Quads[q],
				       gradRhoSpin1Quads[q],
				       excQuads[q],
				       derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
				       derExchCorrEnergyWithGradRhoOutSpin1Quads[q],
				       pseudoVLocQuads[q]);

       Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensorSP::getELocWfcEshelbyTensorPeriodicKPoints
							 (psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
                                                          psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
							  gradPsiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
							  gradPsiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
							  dftPtr->d_kPointCoordinates,
							  dftPtr->d_kPointWeights,
							  dftPtr->eigenValues,
							  dftPtr->fermiEnergy,
							  dftParameters::TVal);

       EKPoints+=eshelbyTensorSP::getEKStress
					 (psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
					  psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
					  gradPsiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
					  gradPsiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
					  dftPtr->d_kPointCoordinates,
					  dftPtr->d_kPointWeights,
					  dftPtr->eigenValues,
					  dftPtr->fermiEnergy,
					  dftParameters::TVal);

       if(isPseudopotential && !dftParameters::useHigherQuadNLP)
       {
           EKPoints+=eshelbyTensorSP::getEnlEshelbyTensorPeriodic
	                                                (ZetaDeltaVQuads[q],
		                                         projectorKetTimesPsiSpin0TimesV,
		                                         projectorKetTimesPsiSpin1TimesV,
						         psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
						         psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
							 dftPtr->d_kPointWeights,
						         dftPtr->eigenValues,
						         dftPtr->fermiEnergy,
						         dftParameters::TVal);

           EKPoints+=eshelbyTensorSP::getEnlStress(gradZetalmDeltaVlDyadicDistImageAtomsQuads[q],
		                                 projectorKetTimesPsiSpin0TimesV,
						 projectorKetTimesPsiSpin1TimesV,
					         psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
					         psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
					         dftPtr->d_kPointWeights,
						 dftPtr->eigenValues,
						 dftPtr->fermiEnergy,
						 dftParameters::TVal);

       }//is pseudopotential check

       EQuadSum+=E*forceEval.JxW(q);
       EKPointsQuadSum+=EKPoints*forceEval.JxW(q);

    }//quad point loop

    if (isPseudopotential && dftParameters::useHigherQuadNLP)
	for (unsigned int q=0; q<numQuadPointsNLP; ++q)
	{
	   Tensor<2,C_DIM,VectorizedArray<double> > EKPoints
                 =eshelbyTensorSP::getEnlEshelbyTensorPeriodic
	                                                (ZetaDeltaVQuads[q],
		                                         projectorKetTimesPsiSpin0TimesV,
		                                         projectorKetTimesPsiSpin1TimesV,
						         psiSpin0QuadsNLP.begin()+q*numEigenVectors*numKPoints,
						         psiSpin1QuadsNLP.begin()+q*numEigenVectors*numKPoints,
							 dftPtr->d_kPointWeights,
						         dftPtr->eigenValues,
						         dftPtr->fermiEnergy,
						         dftParameters::TVal);

           EKPoints+=eshelbyTensorSP::getEnlStress(gradZetalmDeltaVlDyadicDistImageAtomsQuads[q],
		                                 projectorKetTimesPsiSpin0TimesV,
						 projectorKetTimesPsiSpin1TimesV,
					         psiSpin0QuadsNLP.begin()+q*numEigenVectors*numKPoints,
					         psiSpin1QuadsNLP.begin()+q*numEigenVectors*numKPoints,
					         dftPtr->d_kPointWeights,
						 dftPtr->eigenValues,
						 dftPtr->fermiEnergy,
						 dftParameters::TVal);


	   EKPointsQuadSum+=EKPoints*forceEvalNLP.JxW(q);
	}

   for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	for (unsigned int idim=0; idim<C_DIM; ++idim)
	    for (unsigned int jdim=0; jdim<C_DIM; ++jdim)
	    {
		d_stress[idim][jdim]+=EQuadSum[idim][jdim][iSubCell];
		d_stressKPoints[idim][jdim]+=EKPointsQuadSum[idim][jdim][iSubCell];
	    }
  }
}
#endif
