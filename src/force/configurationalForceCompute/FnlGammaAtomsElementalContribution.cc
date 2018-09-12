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
//(locally used function) compute Fnl contibution due to Gamma(Rj) for given set of cells
template<unsigned int FEOrder>
void forceClass<FEOrder>::FnlGammaAtomsElementalContributionPeriodic(std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
							              FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
								      FEEvaluation<C_DIM,1,C_num1DQuadPSP<FEOrder>(),C_DIM>  & forceEvalNLP,
							              const unsigned int cell,
							              const std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > > & pspnlGammaAtomsQuads,
                                                                      const std::vector<std::vector<std::vector<std::complex<double> > > > & projectorKetTimesPsiTimesV,
							              const std::vector<Tensor<1,2,VectorizedArray<double> > > & psiQuads)
{

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numEigenVectors=dftPtr->numEigenValues;
  const unsigned int numKPoints=dftPtr->d_kPointWeights.size();
  const unsigned int numSubCells= dftPtr->matrix_free_data.n_components_filled(cell);
  const unsigned int numQuadPoints=dftParameters::useHigherQuadNLP?
                                   forceEvalNLP.n_q_points
				   :forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  const unsigned int numNonLocalAtomsCurrentProcess= dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

  for(int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
  {
      //
      //get the global charge Id of the current nonlocal atom
      //
      const int nonLocalAtomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int globalChargeIdNonLocalAtom =  dftPtr->d_nonLocalAtomGlobalChargeIds[nonLocalAtomId];
      std::vector<std::vector<std::vector<std::complex<double> > > >   temp2(numKPoints);
      for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
      {
	  temp2[ikPoint].resize(1);
	  temp2[ikPoint][0]=projectorKetTimesPsiTimesV[ikPoint][iAtom];

      }

      if (dftParameters::useHigherQuadNLP)
      {
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	       std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > temp1(1);
	       temp1[0]=pspnlGammaAtomsQuads[q][iAtom];

	       const Tensor<1,C_DIM,VectorizedArray<double> >
		   F=-eshelbyTensor::getFnlPeriodic(temp1,
						    temp2,
						    psiQuads.begin()+q*numEigenVectors*numKPoints,
						    dftPtr->d_kPointWeights,
						    dftPtr->eigenValues,
						    dftPtr->fermiEnergy,
						    dftParameters::TVal);


	       forceEvalNLP.submit_value(F,q);
	  }
      }
      else
      {
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	       std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > temp1(1);
	       temp1[0]=pspnlGammaAtomsQuads[q][iAtom];

	       const Tensor<1,C_DIM,VectorizedArray<double> >
		   F=-eshelbyTensor::getFnlPeriodic(temp1,
						    temp2,
						    psiQuads.begin()+q*numEigenVectors*numKPoints,
						    dftPtr->d_kPointWeights,
						    dftPtr->eigenValues,
						    dftPtr->fermiEnergy,
						    dftParameters::TVal);


	       forceEval.submit_value(F,q);
	  }
      }

      const Tensor<1,C_DIM,VectorizedArray<double> > forceContributionFnlGammaiAtomCells
						     =dftParameters::useHigherQuadNLP?
						      forceEvalNLP.integrate_value()
						      :forceEval.integrate_value();

      if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom)==forceContributionFnlGammaAtoms.end())
	   forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]=std::vector<double>(C_DIM,0.0);

      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	   for (unsigned int idim=0; idim<C_DIM; idim++)
	     forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom][idim]+=
		   forceContributionFnlGammaiAtomCells[idim][iSubCell];
  }//iAtom loop
}

#else

template<unsigned int FEOrder>
void forceClass<FEOrder>::FnlGammaAtomsElementalContributionNonPeriodic(std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
							                FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
								        FEEvaluation<C_DIM,1,C_num1DQuadPSP<FEOrder>(),C_DIM>  & forceEvalNLP,
							                const unsigned int cell,
							                const std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > pspnlGammaAtomQuads,
                                                                        const std::vector<std::vector<double> >  & projectorKetTimesPsiTimesV,
							                const std::vector< VectorizedArray<double> > & psiQuads)
{

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numEigenVectors=dftPtr->numEigenValues;
  const unsigned int numSubCells= dftPtr->matrix_free_data.n_components_filled(cell);
  const unsigned int numQuadPoints=dftParameters::useHigherQuadNLP?
                                   forceEvalNLP.n_q_points
				   :forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  const unsigned int numNonLocalAtomsCurrentProcess= dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

  for(int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
  {
      //
      //get the global charge Id of the current nonlocal atom
      //
      const int nonLocalAtomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int globalChargeIdNonLocalAtom =  dftPtr->d_nonLocalAtomGlobalChargeIds[nonLocalAtomId];
      std::vector<std::vector<double> >  temp2(1);
      temp2[0]=projectorKetTimesPsiTimesV[iAtom];


      if (dftParameters::useHigherQuadNLP)
      {
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	       std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > temp1(1);
	       temp1[0]=pspnlGammaAtomQuads[q][iAtom];

	       const Tensor<1,C_DIM,VectorizedArray<double> > F=
			      -eshelbyTensor::getFnlNonPeriodic(temp1,
								temp2,
								psiQuads.begin()+q*numEigenVectors,
								(dftPtr->eigenValues)[0],
								dftPtr->fermiEnergy,
								dftParameters::TVal);


	       forceEvalNLP.submit_value(F,q);
	  }
      }
      else
      {
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	       std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > temp1(1);
	       temp1[0]=pspnlGammaAtomQuads[q][iAtom];

	       const Tensor<1,C_DIM,VectorizedArray<double> > F=
			      -eshelbyTensor::getFnlNonPeriodic(temp1,
								temp2,
								psiQuads.begin()+q*numEigenVectors,
								(dftPtr->eigenValues)[0],
								dftPtr->fermiEnergy,
								dftParameters::TVal);


	       forceEval.submit_value(F,q);
	  }
      }

      const Tensor<1,C_DIM,VectorizedArray<double> > forceContributionFnlGammaiAtomCells
						     =dftParameters::useHigherQuadNLP?
						      forceEvalNLP.integrate_value()
						      :forceEval.integrate_value();

      if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom)==forceContributionFnlGammaAtoms.end())
	   forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]=std::vector<double>(C_DIM,0.0);

      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	   for (unsigned int idim=0; idim<C_DIM; idim++)
	     forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom][idim]+=
		   forceContributionFnlGammaiAtomCells[idim][iSubCell];

  }//iAtom loop
}
#endif

//(locally used function) accumulate and distribute Fnl contibution due to Gamma(Rj)
template<unsigned int FEOrder>
void forceClass<FEOrder>::distributeForceContributionFnlGammaAtoms(const std::map<unsigned int,std::vector<double> > & forceContributionFnlGammaAtoms)
{
    for (unsigned int iAtom=0;iAtom <dftPtr->atomLocations.size(); iAtom++)
    {

      bool doesAtomIdExistOnLocallyOwnedNode=false;
      if (d_atomsForceDofs.find(std::pair<unsigned int,unsigned int>(iAtom,0))!=d_atomsForceDofs.end())
        doesAtomIdExistOnLocallyOwnedNode=true;

      std::vector<double> forceContributionFnlGammaiAtomGlobal(C_DIM);
      std::vector<double> forceContributionFnlGammaiAtomLocal(C_DIM,0.0);

      if (forceContributionFnlGammaAtoms.find(iAtom)!=forceContributionFnlGammaAtoms.end())
	 forceContributionFnlGammaiAtomLocal=forceContributionFnlGammaAtoms.find(iAtom)->second;
      // accumulate value
      MPI_Allreduce(&(forceContributionFnlGammaiAtomLocal[0]),
		  &(forceContributionFnlGammaiAtomGlobal[0]),
		  3,
		  MPI_DOUBLE,
		  MPI_SUM,
		  mpi_communicator);

      if (doesAtomIdExistOnLocallyOwnedNode)
      {
        std::vector<types::global_dof_index> forceLocalDofIndices(C_DIM);
        for (unsigned int idim=0; idim<C_DIM; idim++)
	    forceLocalDofIndices[idim]=d_atomsForceDofs[std::pair<unsigned int,unsigned int>(iAtom,idim)];
#ifdef USE_COMPLEX
        d_constraintsNoneForce.distribute_local_to_global(forceContributionFnlGammaiAtomGlobal,forceLocalDofIndices,d_configForceVectorLinFEKPoints);
#else
        d_constraintsNoneForce.distribute_local_to_global(forceContributionFnlGammaiAtomGlobal,forceLocalDofIndices,d_configForceVectorLinFE);
#endif
      }
    }
}
