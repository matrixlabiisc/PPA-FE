	    void
    writeOrbitalDataIntoFile(const std::vector<std::vector<int>> &data,
                      const std::string &                     fileName)
    {
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {

          std::ofstream outFile(fileName);
          if (outFile.is_open())
            {
              for (unsigned int irow = 0; irow < data.size(); ++irow)
                {
                  for (unsigned int icol = 0; icol < data[irow].size(); ++icol)
                    {
                      outFile <<data[irow][icol];
                      if (icol < data[irow].size() - 1)
                        outFile << " ";
                    }
                  outFile << "\n";
                }

              outFile.close();
            }
        }
    }
	
	
	
	
	
	
	
	
	void
    readBasisFile(const unsigned int                numColumns,
             std::vector<std::vector<int>> &data,
             const std::string &               fileName)
    {
      std::vector<int> rowData(numColumns, 0.0);
      std::ifstream       readFile(fileName.c_str());
      if (readFile.fail())
        {
          std::cerr << "Error opening file: " << fileName.c_str() << std::endl;
          exit(-1);
        }

      //
      // String to store line and word
      //
      std::string readLine;
      std::string word;

      //
      // column index
      //
      int columnCount;

      if (readFile.is_open())
        {
          while (std::getline(readFile, readLine))
            {
              std::istringstream iss(readLine);

              columnCount = 0;

              while (iss >> word && columnCount < numColumns)
                rowData[columnCount++] = atoi(word.c_str());

              data.push_back(rowData);
            }
        }
      readFile.close();
    }








// using this class we create an Array of Objects
// // each element corresponding to an atom type
// // corresponding atomPositions should be as an external array or suitable in a datastructure

void constructQuantumNumbersHierarchy
		(unsigned int n, unsigned int l, 
		 std::vector<int>& rank)
{
     // assume the vector of size 0 has already been reserved with space for N shells 
     // which is N(N+1)(2N+1)/6 orbitals for N shells 
     // N is maximum of the principal quantum number over each atomType
     // this function is called just once for the whole program 
  
    	/*  	OrbitalQuantumNumbers orbitalTraverse;

		for(unsigned int n = nstart; n <= nend; ++n) {
			for(unsigned int l = 0; l < n; ++l) {
				for(unsigned int tmp_m = 0; tmp_m <= 2*l; ++tmp_m) {

					orbitalTraverse.n = n;
					orbitalTraverse.l = l;
					orbitalTraverse.m = tmp_m - l;
					quantumNumHierarchy.push_back(orbitalTraverse);
				}
			}

		}
		*/
		rank.clear();
		


}

void appendElemsOfRangeToFile(unsigned int start,
							  unsigned int end,
							  std::string filename){

	std::ofstream outputFile;
	outputFile.open(filename, std::ofstream::out | std::ofstream::app);

	if (outputFile.is_open()) {
		
		for (int i = start; i <= end; ++i)
		{
			outputFile << i << '\n';
		}
	}

	else {

		std::cerr << "Couldn't open " << filename << " file!!" << std::endl;
 		exit(0);
	}

	outputFile.close();
	// it is usually not required to close the file 
}
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::newRhoSpillFactor(const dealii::DoFHandler<3> &dofHandlerOfField,
  const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues,
  const std::map<dealii::CellId, std::vector<double>> *NewrhoQuadValues)
{
  double               Numerator = 0.0;
  double 			   Denominator = 0.0;
  const Quadrature<3> &quadrature_formula =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3>        fe_values(dofHandlerOfField.get_fe(),
                        quadrature_formula,
                        update_JxW_values);
  const unsigned int dofs_per_cell = dofHandlerOfField.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  DoFHandler<3>::active_cell_iterator cell = dofHandlerOfField.begin_active(),
                                      endc = dofHandlerOfField.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          const std::vector<double> &rhoValues =
            (*rhoQuadValues).find(cell->id())->second;
          const std::vector<double> &NewrhoValues =
            (*NewrhoQuadValues).find(cell->id())->second;			
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              Denominator += (rhoValues[q_point] * fe_values.JxW(q_point))*(rhoValues[q_point] * fe_values.JxW(q_point));
			  Numerator += (rhoValues[q_point] - NewrhoValues[q_point])  * fe_values.JxW(q_point)*
			  				(rhoValues[q_point] - NewrhoValues[q_point])  * fe_values.JxW(q_point);
            }
        }
    }
  return(sqrt(Utilities::MPI::sum(Numerator, mpi_communicator))/sqrt(Utilities::MPI::sum(Denominator, mpi_communicator)));



}
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::orbitalOverlapPopulationCompute(const std::vector<std::vector<double> > & eigenValuesInput)
{
	pcout << std::fixed;
	pcout << std::setprecision(8);
	double startTime1, endTime1, startTime2, endTime2, startTime3,endTime3;
	MPI_Barrier(MPI_COMM_WORLD);
	startTime1 = MPI_Wtime();
	pcout << "Started post-processing DFT results to obtain Bonding information..\n";

	// would it be good to replace (unsigned) int with (unsigned) short int?
	// would it be better to replace unsigned short int with uint16_t?

	//**************** Forming data structures of atom info ****************// 

	pcout << "reading input files..\n";

	const unsigned int numOfAtoms = atomLocations.size();
	const unsigned int numOfAtomTypes = atomTypes.size(); // is not more than 120

	// above numOfAtoms and numOfAtomTypes would be taken from DFT-FE run
	// actually even the contents of the coordinates files would be taken from DFT-FE run 

	std::vector<unsigned int> atomicNumVec; // vector of atomic numbers of all atoms 
	atomicNumVec.reserve(numOfAtoms);

	// this is not used in the overlap-population-analysis
	std::vector<unsigned int> valenceElectronsVec; // valence electrons for all atoms 
	valenceElectronsVec.reserve(numOfAtoms);

	// can have std::array here! 
	// as all the coordinates are of fixed length 
	// std::vector<std::vector<double>> atomCoordinates(0, std::vector<double>(3, double{}));
	// we could also try a single std::vector, and exxtract coordinates from it
	// by copying into a std::array or use something like span (but need C++20)

	std::vector<std::array<double, 3>> atomCoordinates(0, std::array<double, 3>{});
	atomCoordinates.reserve(numOfAtoms); 

	std::set<unsigned int> atomTypesSet; // only atom types are stored 
	// this is used since they will be repetition of atom types 
	// we assume that the basis required is unique for each atom type 

	unsigned int a, b;
	double x, y, z, zeta;
	unsigned int count = 0;

	std::string coordinatesFile = "coordinates.inp"; // In DFT-FE format

	std::ifstream atomCoordinatesFile (coordinatesFile);
	if (atomCoordinatesFile.is_open()){
		while(atomCoordinatesFile >> a >> b >> x >> y >> z){

			atomicNumVec.push_back(a);//Atom Number of each globalCharge
			atomTypesSet.insert(a); // atom type is determined by the Atomic number 
			valenceElectronsVec.push_back(b);
			atomCoordinates.push_back({x, y, z});

			++count;
		}
	}

	else {

		std::cerr << "Couldn't open " << coordinatesFile << " file!!" << std::endl;
        exit(0);
	}

	atomCoordinatesFile.close();

	pcout << "reading " << coordinatesFile << " complete!\n";

	assert(count == numOfAtoms); // we can add an assert message later 
	assert(numOfAtomTypes == atomTypesSet.size());

	std::vector<unsigned int> atomTypesVec {atomTypesSet.begin(), atomTypesSet.end()}; //Converting set to vector

	// we can delete the atomTypesSet using clear() if needed, later after diff!  

	// atomType to atomTypeID for reference to the basis objects vector 
	// observe that atomID and atomTypeID are different 

	count = 0;
	std::map<unsigned int, unsigned int> atomTypetoAtomTypeID; // a reverse mapping 
	for (auto i : atomTypesVec){

		atomTypetoAtomTypeID.insert({i, count});
		++count; 
	}

	// extracting the num of basis for each atomType and the corresponding zeta values 

	count = 0;
	std::map<unsigned int, unsigned int> atomTypetoBasisDim;
	std::map<unsigned int, unsigned int> atomTypetoNstart;
	std::map<unsigned int, unsigned int> atomTypetoNend;
	std::map<unsigned int, double> atomTypetoZeta;
	std::set<unsigned int> atomTypeswithBasisInfo;
	std::set<unsigned int> diff; // to check if the required data is provided or not
	std::map<unsigned int, unsigned int> atomTypeToBasisInfoStartNum; 

	unsigned int maxBasisShell = 0; // some lower unreachable value 
	// unsigned int minBasisShell = 10; // some unreachable value

	unsigned int tmpBasisDim, b1, b2;

	// b1 and b2 are the range of principal quantum numbers, ends inclusive
	// where the slater type basis are constructed based on n, l, m values or each orbital 


	unsigned int basisHierarchyStart, basisHierarchyEnd, basisCount = 1;

	std::string basisInfoFile = "BasisInfo.inp";



	std::vector<AtomicOrbitalBasisManager> atomTypewiseSTOvector;
	//atomTypewiseSTOvector.reserve(numOfAtomTypes);
	int atomType;
	std::vector<std::vector<int>> atomTypesorbitals;
	readBasisFile(3,atomTypesorbitals,"BasisInfo.inp");


	for(unsigned int i = 0; i < numOfAtomTypes; ++i)
	{
		atomType = atomTypesVec[i];
		atomTypewiseSTOvector.push_back(AtomicOrbitalBasisManager(atomType, d_dftParamsPtr->AtomicOrbitalBasis, true));
	}
	/*** constructQuantumNumbersHierarchy */
	std::vector<std::vector<int>> atomTypewiseOrbitalist;
	std::vector<bool> atomTypeflag(numOfAtomTypes,false);
	std::vector<int> atomTypeoritalstart(numOfAtomTypes,0);
	int counter = 1;
	for (int i = 0; i < atomTypesorbitals.size(); i++)
	{
		for (int j = 0; j <atomTypewiseSTOvector.size(); j++ )
		{
			if (atomTypewiseSTOvector[j].atomType == atomTypesorbitals[i][0] )
			{
				if(atomTypeflag[j] == false )
				{	
					atomTypeoritalstart[j]=counter;
					atomTypeflag[j] = true;

				}
				int n,l,m;
				n = atomTypesorbitals[i][1];
				l = atomTypesorbitals[i][2];
				
				for(m = -l;m <=l; m++)
				{	std::vector<int> tempvec(3,0);
					atomTypewiseSTOvector[j].n.push_back(n);
					atomTypewiseSTOvector[j].l.push_back(l);
					atomTypewiseSTOvector[j].m.push_back(m);
					tempvec[0] = n;
					tempvec[1] = l;
					tempvec[2] = m;
					atomTypewiseOrbitalist.push_back(tempvec);
					counter++;
				}

			
			
			}
		}
		writeOrbitalDataIntoFile(atomTypewiseOrbitalist,"atomTypeWiseOrbitalNums.txt");
		



	}

	for(int j = 0; j <atomTypewiseSTOvector.size(); j++ )
	{
		atomTypewiseSTOvector[j].CreatePseudoAtomicOrbitalBasis();
	}

	pcout << "vector of objects constructed!\n";




	std::ofstream atomWiseAtomicOrbitalInfoFile("atomWiseAtomicOrbitalInfo.txt");

	if (!atomWiseAtomicOrbitalInfoFile.is_open())
	{
		std::cerr << "Couldn't open " << "atomWiseAtomicOrbitalInfo.txt" << " file!!" << std::endl;
    exit(0);
	}

	unsigned int atomicNum; 

	count = 0; 
	std::vector<unsigned int> atomwiseGlobalbasisNum; // cumulative vector  
	atomwiseGlobalbasisNum.reserve(numOfAtoms + 1);

	atomwiseGlobalbasisNum.push_back(count); // first entry is zero 
	

/* Change the above loop to: */	
	for(int i = 0; i < atomicNumVec.size(); i++)
	{

		for (int j = 0; j < atomTypewiseSTOvector.size(); j++ )
		{
			if(atomicNumVec[i] == atomTypewiseSTOvector[j].atomType)
			{
				count += atomTypewiseSTOvector[j].sizeofbasis();
				atomwiseGlobalbasisNum.push_back(count);		
				
				
				
				break;
			
			
			
			
			}
		}
	}


	unsigned int totalDimOfBasis = count; 

	pcout << "total basis dimension: " << totalDimOfBasis << '\n'
			  << "total number of atoms: " << numOfAtoms << '\n'
			  << "number of atoms types: " << numOfAtomTypes << '\n';

	std::vector<LocalAtomicBasisInfo> globalBasisInfo;
	globalBasisInfo.reserve(totalDimOfBasis);
	unsigned int tmp1, tmp2, tmp3, nstart, basisNstart;

	for(unsigned int i = 0; i < numOfAtoms; ++i) {

		atomicNum = atomicNumVec[i];
		tmp1 = atomTypetoAtomTypeID[ atomicNum ];
		tmp2 =  atomwiseGlobalbasisNum[i];
		tmp3 = atomwiseGlobalbasisNum[i+1];
		//nstart = atomTypetoNstart[ atomicNum ];
		//basisNstart = numOfOrbitalsForShellCount(1, nstart - 1);

		atomWiseAtomicOrbitalInfoFile << atomicNum << " "
																	<< tmp2 + 1 << " "
																	<< tmp3 << " "
																	<<atomTypeoritalstart[tmp1]
																	<< '\n';

		for(unsigned int j = tmp2; j < tmp3; ++j)
		{
			//globalBasisInfo.push_back({i, tmp1, basisNstart + j - tmp2}); 
			//globalBasisInfo.push_back({i, tmp1,atomTypewiseSTOvector[tmp1].n ,atomTypewiseSTOvector[tmp1].l ,atomTypewiseSTOvector[tmp1].m});
			LocalAtomicBasisInfo temp;
			temp.atomID = i;
			temp. atomTypeID=tmp1;
			temp.n=atomTypewiseSTOvector[tmp1].n[j - tmp2];
			temp.l=atomTypewiseSTOvector[tmp1].l[j - tmp2];
			temp.m=atomTypewiseSTOvector[tmp1].m[j - tmp2];
			globalBasisInfo.push_back(temp);	
			//  globalBasisInfo.push_back({i, tmp1,atomTypewiseSTOvector[tmp1].n[j - tmp2] ,atomTypewiseSTOvector[tmp1].l[j - tmp2] ,atomTypewiseSTOvector[tmp1].m[j - tmp2]});	
			//globalBasisInfo.push_back({i, tmp1,atomTypewiseSTOvector[tmp1].n[j - tmp2] ,atomTypewiseSTOvector[tmp1].l[j - tmp2] ,atomTypewiseSTOvector[tmp1].m[j - tmp2]});	
			// i required to get atom position coordinates 
			// atomTypeID is required to construct the basis 
			// basisNum in the quantumNumHierarchy
		}



	}

	pcout << "global basis info constructed!\n";

	



	unsigned int numOfKSOrbitals = d_dftParamsPtr->NumofKSOrbitalsproj; 
	


	pcout << "Number of Kohn-Sham orbitals: " << numOfKSOrbitals << '\n';


	std::vector<double> energyLevelsKS;
	std::vector<double> occupationNum;




        occupationNum.resize(numOfKSOrbitals*(d_dftParamsPtr->spinPolarized==1?2:1));
		pcout<<"Size of Occupance vector"<<occupationNum.size()<<std::endl;

		unsigned int numEigenValues = eigenValuesInput[0].size() / (1 + d_dftParamsPtr->spinPolarized);
		pcout<<"Number of Eigenvalues "<<numEigenValues<<std::endl;
        for(unsigned int iEigen = 0; iEigen < numOfKSOrbitals; ++iEigen)
           {
             if(d_dftParamsPtr->spinPolarized==0)
			 {
			 	occupationNum[iEigen] = dftUtils::getPartialOccupancy(eigenValuesInput[0][iEigen],fermiEnergy,C_kb,d_dftParamsPtr->TVal);
             	//pcout<<occupationNum[iEigen]<<std::endl;
			 }
			 else
			 {
				 occupationNum[iEigen]=dftUtils::getPartialOccupancy(eigenValuesInput[0][iEigen],fermiEnergy,C_kb,d_dftParamsPtr->TVal);
				 occupationNum[iEigen+numOfKSOrbitals]=dftUtils::getPartialOccupancy(eigenValuesInput[0][iEigen+numEigenValues],fermiEnergy,C_kb,d_dftParamsPtr->TVal);
			 }
           }


	// Loop over atomic orbitals to evaluate at all nodal points
	
	const IndexSet &locallyOwnedSet = dofHandlerEigen.locally_owned_dofs();
	std::vector<IndexSet::size_type> locallyOwnedDOFs;
	locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
	unsigned int n_dofs = locallyOwnedDOFs.size();
	MPI_Barrier(MPI_COMM_WORLD);
	//std::cout<<"Processor ID: "<<this_mpi_process<<" has dofs total: "<<n_dofs<<std::endl;
	std::vector<double> scaledOrbitalValues_FEnodes(n_dofs * totalDimOfBasis, 0.0);
	std::vector<double> scaledKSOrbitalValues_FEnodes((n_dofs * numOfKSOrbitals)*(d_dftParamsPtr->spinPolarized?0:1), 0.0);
	std::vector<double> scaledKSOrbitalValues_FEnodes_spinup((n_dofs * numOfKSOrbitals)*(d_dftParamsPtr->spinPolarized?1:0), 0.0);
	std::vector<double> scaledKSOrbitalValues_FEnodes_spindown((n_dofs * numOfKSOrbitals)*(d_dftParamsPtr->spinPolarized?1:0), 0.0);
	MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

#ifdef USE_COMPLEX

#else



	for (unsigned int dof = 0; dof < n_dofs; ++dof)
	  {
	    // get nodeID 
	    const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
		if (!constraintsNone.is_constrained(dofID))
		{
	    // get coordinates of the finite-element node
	    	Point<3> node  = d_supportPointsEigen[dofID];

	    	auto count1 = totalDimOfBasis*dof;

	    	for (unsigned int i = 0; i < totalDimOfBasis; ++i)
	      	{
				auto atomTypeID = globalBasisInfo[i].atomTypeID;
				auto atomChargeID =globalBasisInfo[i].atomID;
                
				  std::vector<int> imageIdsList;
                  if (d_dftParamsPtr->periodicX ||d_dftParamsPtr->periodicY ||
                      d_dftParamsPtr->periodicZ)
                    {
                      imageIdsList = d_globalChargeIdToImageIdMap[atomChargeID];
                    }
                  else
                    {
                      imageIdsList.push_back(atomChargeID);
                    } 

				for(int imageID = 0; imageID <imageIdsList.size(); imageID++)
				{
					int chargeId = imageIdsList[imageID];
					std::vector<double> atomPos(3,0.0);
                    if (chargeId < numOfAtoms)
                        {
                          atomPos[0] = atomLocations[chargeId][2];
                          atomPos[1] = atomLocations[chargeId][3];
                          atomPos[2] = atomLocations[chargeId][4];
                        }
                    else
                        {
                          atomPos[0] =
                            d_imagePositions[chargeId - numOfAtoms][0];
                          atomPos[1] =
                            d_imagePositions[chargeId - numOfAtoms][1];
                          atomPos[2] =
                            d_imagePositions[chargeId - numOfAtoms][2];
                        }								

					OrbitalQuantumNumbers orbital= {globalBasisInfo[i].n, globalBasisInfo[i].l,globalBasisInfo[i].m};

		  			if(d_dftParamsPtr->AtomicOrbitalBasis == 1)
            		{     
						scaledOrbitalValues_FEnodes[count1 + i] += d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                                   atomTypewiseSTOvector[atomTypeID].bungeOrbital
                                                    (orbital, node, atomPos);
							
					}										
					if(d_dftParamsPtr->AtomicOrbitalBasis == 0)
					{
                 		scaledOrbitalValues_FEnodes[count1 + i] += d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                                   atomTypewiseSTOvector[atomTypeID].PseudoAtomicOrbitalvalue
                                                    (orbital, node, atomPos);

                              									
					}
			  	}			

	      	}
		}		
	    	auto count2 = numOfKSOrbitals*dof;

	    	for (unsigned int j = 0; j < numOfKSOrbitals; ++j)
	      		{

					if(d_dftParamsPtr->spinPolarized == 1)
					{
						scaledKSOrbitalValues_FEnodes_spinup[count2 + j] =  d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) * 
																	 d_eigenVectorsFlattenedSTL[0][dof * d_numEigenValues + j];	
						//pcout<<"Accessing spin down"<<std::endl;											 
						scaledKSOrbitalValues_FEnodes_spindown[count2 + j] =  d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) * 
																	 d_eigenVectorsFlattenedSTL[1][dof * d_numEigenValues + j];	
											 																 

					}	
					else
					{
					
						scaledKSOrbitalValues_FEnodes[count2 + j] =  d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) * 
																d_eigenVectorsFlattenedSTL[0][dof * d_numEigenValues + j];	
						
					}
		 
		  		}
  
		
	  }
	 

#endif
	MPI_Barrier(MPI_COMM_WORLD);
	pcout << "matrices of orbital values at the nodes constructed!\n";
	pcout<<"Time to construct phi and psi matrices: "<<MPI_Wtime() - t1<<std::endl;

		if(this_mpi_process == 0)
		{
		// and writing the high level basis information  

  			std::ofstream highLevelBasisInfoFile ("highLevelBasisInfo.txt");	

			if (highLevelBasisInfoFile.is_open())
			{
				highLevelBasisInfoFile << numOfAtoms << '\n'
													 << numOfAtomTypes << '\n'
													 << totalDimOfBasis << '\n'
													 << numOfKSOrbitals << '\n';

				highLevelBasisInfoFile.close();
			}

			else 
			pcout << "couldn't open highLevelBasisInfo.txt file!\n";


		}
	MPI_Barrier(MPI_COMM_WORLD);


		// direct assembly of Overlap matrix S using Mass diagonal matrix from Gauss Lobatto

		// actually we must write such that only the symmetric upper triangular part is calculated
		// observe in the above we used same variable at two arguments in the function
		// and this function have those arguments in const referenced way! Let's see if it works.. 
	
	
		// COOP Analysis Begin
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		auto upperTriaOfSserial = selfMatrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis);
		std::vector<double> upperTriaOfS((totalDimOfBasis*(totalDimOfBasis+1)/2),0.0);
		// Use MPI_all reduce to get S contribution from other procs.
    	MPI_Allreduce(&upperTriaOfSserial[0],
                          &upperTriaOfS[0],
                          (totalDimOfBasis*(totalDimOfBasis+1)/2),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);				  
		pcout<<"Computation Time of S matrix: "<<MPI_Wtime()-t1;
		pcout<< "Upper triangular part of Overlap matrix (S) vector in the direct way: \n";
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorToFile(upperTriaOfS, "overlapMatrix.txt");
			printVector(upperTriaOfS);
		} 		
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	auto invS = inverseOfOverlapMatrix(upperTriaOfS, totalDimOfBasis);
	MPI_Barrier(MPI_COMM_WORLD);
	endTime1 = MPI_Wtime();
	pcout<<"Computation Time of S inverse: "<<MPI_Wtime()-t1<<std::endl;

	if(d_dftParamsPtr->spinPolarized == 1)
	{	
	
 		auto arrayVecOfProjserial_spinup = matrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   				 scaledKSOrbitalValues_FEnodes_spinup, n_dofs, numOfKSOrbitals);
		std::vector<double> arrayVecOfProj_spinup(totalDimOfBasis*numOfKSOrbitals,0.0);
    	MPI_Allreduce(&arrayVecOfProjserial_spinup[0],
                          &arrayVecOfProj_spinup[0],
                          (totalDimOfBasis*numOfKSOrbitals),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);
 		auto arrayVecOfProjserial_spindown = matrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   				 scaledKSOrbitalValues_FEnodes_spindown, n_dofs, numOfKSOrbitals);
		std::vector<double> arrayVecOfProj_spindown(totalDimOfBasis*numOfKSOrbitals,0.0);
    	MPI_Allreduce(&arrayVecOfProjserial_spindown[0],
                          &arrayVecOfProj_spindown[0],
                          (totalDimOfBasis*numOfKSOrbitals),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);				  
				  
		pcout << "Matrix of projections with atomic orbitals: \n";
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
			{	
				writeVectorAs2DMatrix(arrayVecOfProj_spinup, totalDimOfBasis, numOfKSOrbitals,
												"projOfKSOrbitalsWithAOs_spinup.txt");
				printVector(arrayVecOfProj_spinup);
				pcout<<std::endl;
				writeVectorAs2DMatrix(arrayVecOfProj_spindown, totalDimOfBasis, numOfKSOrbitals,
												"projOfKSOrbitalsWithAOs_spindown.txt");
				printVector(arrayVecOfProj_spindown);
				pcout<<std::endl;				
				pcout<< "Full S inverse matrix: \n";
				printVector(invS);
			}											
	
	
		auto coeffArrayVecOfProj_spinup
					= matrixmatrixmul(invS, 		  totalDimOfBasis, totalDimOfBasis, 
									  				arrayVecOfProj_spinup, totalDimOfBasis, numOfKSOrbitals);
		auto coeffArrayVecOfProj_spindown
					= matrixmatrixmul(invS, 		  totalDimOfBasis, totalDimOfBasis, 
									  				arrayVecOfProj_spindown, totalDimOfBasis, numOfKSOrbitals);													  

		pcout << "Matrix of coefficients of projections: \n";
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorAs2DMatrix(coeffArrayVecOfProj_spinup, totalDimOfBasis, numOfKSOrbitals,
												"coeffsOfKSOrbitalsProjOnAOs_spinup.txt");
			//printVector(coeffArrayVecOfProj_spinup);
			pcout<<std::endl;
			writeVectorAs2DMatrix(coeffArrayVecOfProj_spindown, totalDimOfBasis, numOfKSOrbitals,
												"coeffsOfKSOrbitalsProjOnAOs_spindown.txt");
			//printVector(coeffArrayVecOfProj_spindown);			
		}											
	
	
		std::vector<double> CoeffofOrthonormalisedKSonAO_spinup = OrthonormalizationofProjectedWavefn(upperTriaOfS,totalDimOfBasis, totalDimOfBasis,
														coeffArrayVecOfProj_spinup,totalDimOfBasis, numOfKSOrbitals);	
		std::vector<double> CoeffofOrthonormalisedKSonAO_spindown = OrthonormalizationofProjectedWavefn(upperTriaOfS,totalDimOfBasis, totalDimOfBasis,
														coeffArrayVecOfProj_spindown,totalDimOfBasis, numOfKSOrbitals);														
		pcout<<"C bar Output:"<<std::endl;
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{	
			writeVectorAs2DMatrix(CoeffofOrthonormalisedKSonAO_spinup, totalDimOfBasis, numOfKSOrbitals,
												"OrthocoeffsOfKSOrbitalsProjOnAOsCOOP_spinup.txt");																							
			//printVector(CoeffofOrthonormalisedKSonAO_spinup);
			pcout<<std::endl;
			writeVectorAs2DMatrix(CoeffofOrthonormalisedKSonAO_spindown, totalDimOfBasis, numOfKSOrbitals,
												"OrthocoeffsOfKSOrbitalsProjOnAOsCOOP_spindown.txt");																							
			//printVector(CoeffofOrthonormalisedKSonAO_spindown);			
		}	

		//COOP Analysis End
		pcout<<"--------------------------COOP Data Saved------------------------------"<<std::endl;
	
		//COHP Analysis Begin	

		auto OrthoscaledOrbitalValues_FEnodes = LowdenOrtho(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis,upperTriaOfS);	

		auto upperTriaOfOrthoSserial = selfMatrixTmatrixmul(OrthoscaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis);
		std::vector<double> upperTriaOfOrthoS(totalDimOfBasis*(totalDimOfBasis+1)/2,0.0);
    	MPI_Allreduce(&upperTriaOfOrthoSserial[0],
                          &upperTriaOfOrthoS[0],
                          (totalDimOfBasis*(totalDimOfBasis+1))/2,
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);	
		pcout << "Upper triangular part of Overlap matrix (S) vector in the direct way: \n";
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorToFile(upperTriaOfOrthoS, "OrthooverlapMatrix.txt");
			//printVector(upperTriaOfOrthoS);
		}	
	
 		auto coeffarrayVecOfOrthoProjserial_spinup = matrixTmatrixmul(OrthoscaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   				 scaledKSOrbitalValues_FEnodes_spinup, n_dofs, numOfKSOrbitals);
	
		std::vector<double> coeffarrayVecOfOrthoProj_spinup(totalDimOfBasis*numOfKSOrbitals,0.0);
    	MPI_Allreduce(&coeffarrayVecOfOrthoProjserial_spinup[0],
                          &coeffarrayVecOfOrthoProj_spinup[0],
                          (totalDimOfBasis*numOfKSOrbitals),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);	
 		auto coeffarrayVecOfOrthoProjserial_spindown = matrixTmatrixmul(OrthoscaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   				 scaledKSOrbitalValues_FEnodes_spindown, n_dofs, numOfKSOrbitals);
	
		std::vector<double> coeffarrayVecOfOrthoProj_spindown(totalDimOfBasis*numOfKSOrbitals,0.0);
    	MPI_Allreduce(&coeffarrayVecOfOrthoProjserial_spindown[0],
                          &coeffarrayVecOfOrthoProj_spindown[0],
                          (totalDimOfBasis*numOfKSOrbitals),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);						  
					  
//		pcout << "Matrix of projections with Ortho atomic orbitals: \n";

		pcout << "Matrix of coefficients of projections: \n";
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorAs2DMatrix(coeffarrayVecOfOrthoProj_spinup, totalDimOfBasis, numOfKSOrbitals,
												"coeffsOfKSOrbitalsProjOnAOsforCOHP_spinup.txt");
			//printVector(coeffarrayVecOfOrthoProj_spinup);
			pcout<<std::endl;
			writeVectorAs2DMatrix(coeffarrayVecOfOrthoProj_spindown, totalDimOfBasis, numOfKSOrbitals,
												"coeffsOfKSOrbitalsProjOnAOsforCOHP_spindown.txt");
			//printVector(coeffarrayVecOfOrthoProj_spindown);			

		}											
		std::vector<double> CoeffofOrthonormalisedKSonAO_COHP_spinup = OrthonormalizationofProjectedWavefn(upperTriaOfOrthoS,totalDimOfBasis, totalDimOfBasis,
														coeffarrayVecOfOrthoProj_spinup,totalDimOfBasis, numOfKSOrbitals);
		std::vector<double> CoeffofOrthonormalisedKSonAO_COHP_spindown = OrthonormalizationofProjectedWavefn(upperTriaOfOrthoS,totalDimOfBasis, totalDimOfBasis,
														coeffarrayVecOfOrthoProj_spindown,totalDimOfBasis, numOfKSOrbitals);															
		pcout<<"C hat Output:"<<std::endl;
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorAs2DMatrix(CoeffofOrthonormalisedKSonAO_COHP_spinup, totalDimOfBasis, numOfKSOrbitals,
												"OrthocoeffsOfKSOrbitalsProjOnAOsCOHP_spinup.txt");
			//printVector(CoeffofOrthonormalisedKSonAO_COHP_spinup);
			pcout<<std::endl;
			writeVectorAs2DMatrix(CoeffofOrthonormalisedKSonAO_COHP_spindown, totalDimOfBasis, numOfKSOrbitals,
												"OrthocoeffsOfKSOrbitalsProjOnAOsCOHP_spindown.txt");
			//printVector(CoeffofOrthonormalisedKSonAO_COHP_spindown);			
		}											

		//Compute projected Hamiltonian of FE discretized Hamiltonian into 
		pcout<<"--------------------------COHP Data Saved------------------------------"<<std::endl;
		if(this_mpi_process == 0)
		{					
		// writing the energy levels and the occupation numbers 
  			unsigned int kPointDummy = 0;
 
			std::ofstream energyLevelsOccNumsFile ("energyLevelsOccNums.txt");

			if (energyLevelsOccNumsFile.is_open())
			{
				for (unsigned int i = 0; i < eigenValues[0].size(); ++i)
				{
					const double partialOccupancy = dftUtils::getPartialOccupancy(
                              eigenValues[kPointDummy][i], fermiEnergy, C_kb, d_dftParamsPtr->TVal);
			
					energyLevelsOccNumsFile << eigenValues[kPointDummy][i]
                              << " " << partialOccupancy << '\n';
				}

				energyLevelsOccNumsFile.close();
			}

			else 
			pcout << "couldn't open energyLevelsOccNums.txt file!\n";





			pcout<<"\n-------------------------------------------------------\n";
			pcout<<"Projected SpillFactors are:"<<std::endl;
			spillFactorsofProjectionwithCS(coeffArrayVecOfProj_spinup,coeffArrayVecOfProj_spindown,upperTriaOfS,occupationNum ,
									totalDimOfBasis, numOfKSOrbitals,
									totalDimOfBasis,totalDimOfBasis	);
			pcout<<"\n-------------------------------------------------------\n";

		}		

	}	
	if(d_dftParamsPtr->spinPolarized == 0)
		{	
			MPI_Barrier(MPI_COMM_WORLD);
			t1 = MPI_Wtime();
			startTime2 = MPI_Wtime();	
 			auto arrayVecOfProjserial = matrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   				 scaledKSOrbitalValues_FEnodes, n_dofs, numOfKSOrbitals);
			std::vector<double> arrayVecOfProj(totalDimOfBasis*numOfKSOrbitals,0.0);
    		MPI_Allreduce(&arrayVecOfProjserial[0],
                          &arrayVecOfProj[0],
                          (totalDimOfBasis*numOfKSOrbitals),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);
				  
			MPI_Barrier(MPI_COMM_WORLD);
			pcout<<"Time of phiTpsi: "<<MPI_Wtime()-t1<<std::endl;		  
			pcout << "Matrix of projections with atomic orbitals: \n";
	
			if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
			{	
				writeVectorAs2DMatrix(arrayVecOfProj, totalDimOfBasis, numOfKSOrbitals,
												"projOfKSOrbitalsWithAOs.txt");
				//printVector(arrayVecOfProj);
				pcout<< "Full S inverse matrix: \n";
				//printVector(invS);
			} 											
	
			MPI_Barrier(MPI_COMM_WORLD);
			t1 = MPI_Wtime();
			auto coeffArrayVecOfProj
					= matrixmatrixmul(invS, 		  totalDimOfBasis, totalDimOfBasis, 
									  				arrayVecOfProj, totalDimOfBasis, numOfKSOrbitals);
			pcout << "Matrix of coefficients of projections: \n";
			/*
			if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
			{
			writeVectorAs2DMatrix(coeffArrayVecOfProj, totalDimOfBasis, numOfKSOrbitals,
												"coeffsOfKSOrbitalsProjOnAOs.txt");
			//printVector(coeffArrayVecOfProj);
			}	 	*/					
			std::vector<double> CoeffofOrthonormalisedKSonAO = OrthonormalizationofProjectedWavefn(upperTriaOfS,totalDimOfBasis, totalDimOfBasis,
														coeffArrayVecOfProj,totalDimOfBasis, numOfKSOrbitals);	
			MPI_Barrier(MPI_COMM_WORLD);
			pcout<<"Time to compute Cbar:"<<MPI_Wtime()-t1<<std::endl;
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{	
			writeVectorAs2DMatrix(CoeffofOrthonormalisedKSonAO, totalDimOfBasis, numOfKSOrbitals,
												"FePOP_Coeff_v1.txt");																							
			//printVector(CoeffofOrthonormalisedKSonAO);
		} 	

		//COOP Analysis End
		pcout<<"--------------------------COOP Data Saved------------------------------"<<std::endl;
	
		//COHP Analysis Begin	
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();		
		auto OrthoscaledOrbitalValues_FEnodes = LowdenOrtho(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis,upperTriaOfS);
		MPI_Barrier(MPI_COMM_WORLD);
		pcout<<"Computation of ortho phi from phi: "<<MPI_Wtime()-t1<<std::endl;	
		auto upperTriaOfOrthoSserial = selfMatrixTmatrixmul(OrthoscaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis);
		std::vector<double> upperTriaOfOrthoS(totalDimOfBasis*(totalDimOfBasis+1)/2,0.0);
    	MPI_Allreduce(&upperTriaOfOrthoSserial[0],
                          &upperTriaOfOrthoS[0],
                          (totalDimOfBasis*(totalDimOfBasis+1))/2,
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);	
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();						  
		pcout << "Upper triangular part of Overlap matrix (S) vector in the direct way: \n";
		/*
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorToFile(upperTriaOfOrthoS, "OrthooverlapMatrix.txt");
			//printVector(upperTriaOfOrthoS);
		} */	
	
 		auto coeffarrayVecOfOrthoProjserial = matrixTmatrixmul(OrthoscaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   				 scaledKSOrbitalValues_FEnodes, n_dofs, numOfKSOrbitals);
	
		std::vector<double> coeffarrayVecOfOrthoProj(totalDimOfBasis*numOfKSOrbitals,0.0);
    	MPI_Allreduce(&coeffarrayVecOfOrthoProjserial[0],
                          &coeffarrayVecOfOrthoProj[0],
                          (totalDimOfBasis*numOfKSOrbitals),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);	
					  
		/*pcout << "Matrix of projections with Ortho atomic orbitals: \n";

		pcout << "Matrix of coefficients of projections: \n";
	
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorAs2DMatrix(coeffarrayVecOfOrthoProj, totalDimOfBasis, numOfKSOrbitals,
												"coeffsOfKSOrbitalsProjOnAOsforCOHP.txt");
			//printVector(coeffarrayVecOfOrthoProj);
		} 	*/									
			std::vector<double> CoeffofOrthonormalisedKSonAO_COHP = OrthonormalizationofProjectedWavefn(upperTriaOfOrthoS,totalDimOfBasis, totalDimOfBasis,
														coeffarrayVecOfOrthoProj,totalDimOfBasis, numOfKSOrbitals);	
			MPI_Barrier(MPI_COMM_WORLD);
			pcout<<"Computation of C_hat:"<<MPI_Wtime()-t1<<std::endl;
	
			if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
			{
			writeVectorAs2DMatrix(CoeffofOrthonormalisedKSonAO_COHP, totalDimOfBasis, numOfKSOrbitals,
												"FePHP_Coeff_v1.txt");
			//printVector(CoeffofOrthonormalisedKSonAO_COHP);
			} 											

		//Compute projected Hamiltonian of FE discretized Hamiltonian into 
			pcout<<"--------------------------COHP Data Saved------------------------------"<<std::endl;		
			MPI_Barrier(MPI_COMM_WORLD);
			endTime2 = MPI_Wtime();
			if(this_mpi_process == 0)
			{					
		// writing the energy levels and the occupation numbers 
  				unsigned int kPointDummy = 0;
				std::ofstream energyLevelsOccNumsFile ("energyLevelsOccNums.txt");

				if (energyLevelsOccNumsFile.is_open())
				{
					for (unsigned int i = 0; i < eigenValues[0].size(); ++i)
						{
						const double partialOccupancy = dftUtils::getPartialOccupancy(
                              eigenValues[kPointDummy][i], fermiEnergy, C_kb, d_dftParamsPtr->TVal);
			
						energyLevelsOccNumsFile << eigenValues[kPointDummy][i]
                              << " " << partialOccupancy << '\n';
						}

					energyLevelsOccNumsFile.close();
				}

				else 
				pcout << "couldn't open energyLevelsOccNums.txt file!\n";
			} 


		pcout<<"--------------------------COHP v2 Starting------------------------------"<<std::endl;
		const unsigned int N = totalDimOfBasis;	
	
#ifdef USE_COMPLEX

#else
		std::vector<dataTypes::number> CoeffNew(N*N,0.0);	 
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	startTime3 = MPI_Wtime();
	 std::vector<dataTypes::number> ProjHam;
	 MPI_Barrier(MPI_COMM_WORLD);
	 d_kohnShamDFTOperatorPtr->XtHX(OrthoscaledOrbitalValues_FEnodes,
					totalDimOfBasis,
					ProjHam);
	MPI_Barrier(MPI_COMM_WORLD);
	pcout<<"Computation of H projected: "<<MPI_Wtime()-t1<<std::endl;				
		if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		{
			writeVectorAs2DMatrix(ProjHam, totalDimOfBasis, totalDimOfBasis,
												"ProjectedHamilton.txt");
			printVector(ProjHam);
			pcout<<std::endl;
		
		} 
			
	 MPI_Barrier(MPI_COMM_WORLD);
	 t1 = MPI_Wtime();	
	  int                info; 
	  
	  
	  if(this_mpi_process == 0)
	  {
      	const unsigned int lwork = 1 + 6*N +
                                 2 * N*N,
                         liwork = 3 + 5 *N;
      	std::vector<int>    iwork(liwork, 0);
      	const char          jobz = 'V', uplo = 'U';
      	std::vector<double> work(lwork);
	  	std::vector<double> D(N,0.0);
      	dftfe::dsyevd_(&jobz,
              &uplo,
              &N,
              &ProjHam[0],
              &N,
              &D[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info); 
      	//
      	// free up memory associated with work
      	//
    	work.clear();
    	iwork.clear();
    	std::vector<double>().swap(work);
    	std::vector<int>().swap(iwork);
		if(info > 0)
			std::cout<<"Eigen Value Decomposition Failed!!"<<std::endl;
		pcout<<"Finished Computation of C_hat2"<<std::endl;
	
		CoeffNew = ProjHam;
	}
    MPI_Allreduce(MPI_IN_PLACE,
                          &CoeffNew[0],
                          N*N,
                          dataTypes::mpi_type_id(&CoeffNew[0]),
                          MPI_SUM,
                          MPI_COMM_WORLD);		
	MPI_Barrier(MPI_COMM_WORLD);
	pcout<<"Computation Time of C_hat new from diagonalization of H: "<<MPI_Wtime()-t1<<std::endl;
	if(this_mpi_process==0)
		writeVectorAs2DMatrix(CoeffNew, totalDimOfBasis, totalDimOfBasis,
												"FePHP_v2.txt");

		

	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();			
	auto phiTphihatserial = matrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   				 OrthoscaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis);
	std::vector<double> phiTphihat(N*N,0.0);											 
    MPI_Allreduce(&phiTphihatserial[0],
                          &phiTphihat[0],
                          (totalDimOfBasis*numOfKSOrbitals),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	pcout<<"Computation of phiTphi_hat: "<<MPI_Wtime()-t1<<std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();						  
	std::vector<double> CoeffNewCOOP(totalDimOfBasis*numOfKSOrbitals, 0.0);

	auto tempmult = matrixmatrixTmul(phiTphihat,totalDimOfBasis,totalDimOfBasis,CoeffNew,totalDimOfBasis,totalDimOfBasis);
	CoeffNewCOOP = matrixmatrixmul(invS,totalDimOfBasis,totalDimOfBasis,tempmult,totalDimOfBasis,totalDimOfBasis);
	MPI_Barrier(MPI_COMM_WORLD);
	endTime3 = MPI_Wtime();
	pcout<<"Computation Time of C_bar for COOP new: "<<MPI_Wtime()-t1<<std::endl;			
	if (this_mpi_process == 0)
	{
		writeVectorAs2DMatrix(CoeffNewCOOP, totalDimOfBasis, totalDimOfBasis,
												"FePOP_v2.txt");
			pcout<<std::endl;
	
	} 					
#endif			

		pcout<<"--------------------------COHP v2 Data Saved-----------------------------"<<std::endl;
		pcout<<"Time1: "<<endTime1-startTime1<<std::endl;
		pcout<<"Time2: "<<endTime2-startTime2<<std::endl;
		pcout<<"Time3: "<<endTime3-startTime3<<std::endl;
		pcout<<" -------------------------New Error Metric--------------------------------"<<std::endl;
#ifdef USE_COMPLEX

#else

	std::vector<double> Minv_scaledOrbitalValues_FEnodes(n_dofs*totalDimOfBasis,0.0);
	for (unsigned int dof = 0; dof < n_dofs; ++dof)
	{
	    // get nodeID 
	    const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
		if (!constraintsNone.is_constrained(dofID))
		{


	    	auto count1 = totalDimOfBasis*dof;

	    	for (unsigned int i = 0; i < totalDimOfBasis; ++i)
	      	{			
    
						Minv_scaledOrbitalValues_FEnodes[count1 + i] += d_kohnShamDFTOperatorPtr->d_invSqrtMassVector.local_element(dof) *
							OrthoscaledOrbitalValues_FEnodes[count1 + i];
							
															

			}			

	    }
		
	}		
 
	auto psiProjected1 = matrixmatrixmul(Minv_scaledOrbitalValues_FEnodes,n_dofs,totalDimOfBasis,
												CoeffofOrthonormalisedKSonAO_COHP,totalDimOfBasis,totalDimOfBasis);

	auto psiProjected2 = matrixmatrixTmul(Minv_scaledOrbitalValues_FEnodes,n_dofs,totalDimOfBasis,
												CoeffNew,totalDimOfBasis,totalDimOfBasis);													
						  
	 	MPI_Barrier(MPI_COMM_WORLD);
		std::vector<std::vector<double>> psi_projected1,psi_projected2;
		psi_projected1.push_back(psiProjected1);
		psi_projected2.push_back(psiProjected2);
		std::map<dealii::CellId, std::vector<double>> rhoValues1, rhoValues2;

  const unsigned int numQuadPoints =
    matrix_free_data.get_n_q_points(d_densityQuadratureId);
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
  												endc = dofHandler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        const dealii::CellId cellId = cell->id();

            (rhoValues1)[cellId]  =
              std::vector<double>(numQuadPoints, 0.0);
            (rhoValues2)[cellId]  =
              std::vector<double>(numQuadPoints, 0.0);
          
      }	

      computeRhoFromPSICPU(
        psi_projected1,
        d_eigenVectorsRotFracDensityFlattenedSTL,
        totalDimOfBasis,
        d_numEigenValuesRR,
        n_dofs,
        eigenValuesInput,
        fermiEnergy,
        fermiEnergyUp,
        fermiEnergyDown,
        *d_kohnShamDFTOperatorPtr,
        dofHandler,
        matrix_free_data.n_physical_cells(),
        matrix_free_data.get_dofs_per_cell(d_densityDofHandlerIndex),
        matrix_free_data.get_quadrature(d_densityQuadratureId).size(),
        d_kPointWeights,
        &rhoValues1,
        gradRhoOutValues,
        rhoOutValuesSpinPolarized,
        gradRhoOutValuesSpinPolarized,
        d_dftParamsPtr->xcFamilyType == "GGA",
        d_mpiCommParent,
        interpoolcomm,
        interBandGroupComm,
        *d_dftParamsPtr,
        true && d_numEigenValues != d_numEigenValuesRR,
        false);		
      computeRhoFromPSICPU(
        psi_projected2,
        d_eigenVectorsRotFracDensityFlattenedSTL,
        totalDimOfBasis,
        d_numEigenValuesRR,
        n_dofs,
        eigenValuesInput,
        fermiEnergy,
        fermiEnergyUp,
        fermiEnergyDown,
        *d_kohnShamDFTOperatorPtr,
        dofHandler,
        matrix_free_data.n_physical_cells(),
        matrix_free_data.get_dofs_per_cell(d_densityDofHandlerIndex),
        matrix_free_data.get_quadrature(d_densityQuadratureId).size(),
        d_kPointWeights,
        &rhoValues2,
        gradRhoOutValues,
        rhoOutValuesSpinPolarized,
        gradRhoOutValuesSpinPolarized,
        d_dftParamsPtr->xcFamilyType == "GGA",
        d_mpiCommParent,
        interpoolcomm,
        interBandGroupComm,
        *d_dftParamsPtr,
        true && d_numEigenValues != d_numEigenValuesRR,
        false); 


		pcout<<"Total Charge old and new "<<totalCharge(d_dofHandlerPRefined,rhoOutValues)<<" "<<totalCharge(d_dofHandlerPRefined,&rhoValues1)<<" "<<totalCharge(d_dofHandlerPRefined,&rhoValues2)<<std::endl;
		pcout<<"New error with old methods= "<<newRhoSpillFactor(d_dofHandlerPRefined, 
											rhoOutValues, &rhoValues1 )<<std::endl;
		pcout<<"New error with new methods= "<<newRhoSpillFactor(d_dofHandlerPRefined, 
											rhoOutValues, &rhoValues2 )<<std::endl;											
		if(this_mpi_process == 0)
		{
			pcout<<"\n-------------------------------------------------------\n";
			pcout<<"Projected SpillFactors are:"<<std::endl;
			spillFactorsofProjectionwithCS(coeffArrayVecOfProj,upperTriaOfS,occupationNum ,
									totalDimOfBasis, numOfKSOrbitals,
									totalDimOfBasis,totalDimOfBasis	);
			pcout<<"\n-------------------------------------------------------\n";

		}
		}

#endif


				



}
