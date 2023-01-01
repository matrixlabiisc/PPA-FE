
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::hamiltonianPopulationCompute(
  const std::vector<std::vector<double>> &eigenValuesInput)
{
  pcout << std::fixed;
  pcout << std::setprecision(8);
  MPI_Barrier(MPI_COMM_WORLD);
  pcout
    << "Started post-processing DFT results to obtain Bonding information..\n";

  // would it be good to replace (unsigned) int with (unsigned) short int?
  // would it be better to replace unsigned short int with uint16_t?

  //**************** Forming data structures of atom info ****************//

  pcout << "reading input files..\n";

  const unsigned int numOfAtoms     = atomLocations.size();
  const unsigned int numOfAtomTypes = atomTypes.size(); // is not more than 120

  // above numOfAtoms and numOfAtomTypes would be taken from DFT-FE run
  // actually even the contents of the coordinates files would be taken from
  // DFT-FE run

  std::vector<unsigned int>
    atomicNumVec; // vector of atomic numbers of all atoms
  atomicNumVec.reserve(numOfAtoms);

  // this is not used in the overlap-population-analysis
  std::vector<unsigned int>
    valenceElectronsVec; // valence electrons for all atoms
  valenceElectronsVec.reserve(numOfAtoms);

  // can have std::array here!
  // as all the coordinates are of fixed length
  // std::vector<std::vector<double>> atomCoordinates(0, std::vector<double>(3,
  // double{})); we could also try a single std::vector, and exxtract
  // coordinates from it by copying into a std::array or use something like span
  // (but need C++20)

  std::vector<std::array<double, 3>> atomCoordinates(0,
                                                     std::array<double, 3>{});
  atomCoordinates.reserve(numOfAtoms);

  std::set<unsigned int> atomTypesSet; // only atom types are stored
  // this is used since they will be repetition of atom types
  // we assume that the basis required is unique for each atom type

  unsigned int a, b;
  double       x, y, z, zeta;
  unsigned int count = 0;

  std::string coordinatesFile = "coordinates.inp"; // In DFT-FE format

  std::ifstream atomCoordinatesFile(coordinatesFile);
  if (atomCoordinatesFile.is_open())
    {
      while (atomCoordinatesFile >> a >> b >> x >> y >> z)
        {
          atomicNumVec.push_back(a); // Atom Number of each globalCharge
          atomTypesSet.insert(
            a); // atom type is determined by the Atomic number
          valenceElectronsVec.push_back(b);
          atomCoordinates.push_back({x, y, z});

          ++count;
        }
    }

  else
    {
      std::cerr << "Couldn't open " << coordinatesFile << " file!!"
                << std::endl;
      exit(0);
    }

  atomCoordinatesFile.close();

  pcout << "reading " << coordinatesFile << " complete!\n";

  assert(count == numOfAtoms); // we can add an assert message later
  assert(numOfAtomTypes == atomTypesSet.size());

  std::vector<unsigned int> atomTypesVec{
    atomTypesSet.begin(), atomTypesSet.end()}; // Converting set to vector

  // we can delete the atomTypesSet using clear() if needed, later after diff!

  // atomType to atomTypeID for reference to the basis objects vector
  // observe that atomID and atomTypeID are different

  count = 0;
  std::map<unsigned int, unsigned int>
    atomTypetoAtomTypeID; // a reverse mapping
  for (auto i : atomTypesVec)
    {
      atomTypetoAtomTypeID.insert({i, count});
      ++count;
    }

  // extracting the num of basis for each atomType and the corresponding zeta
  // values

  count = 0;
  std::map<unsigned int, unsigned int> atomTypetoBasisDim;
  std::map<unsigned int, unsigned int> atomTypetoNstart;
  std::map<unsigned int, unsigned int> atomTypetoNend;
  std::map<unsigned int, double>       atomTypetoZeta;
  std::set<unsigned int>               atomTypeswithBasisInfo;
  std::set<unsigned int>
                                       diff; // to check if the required data is provided or not
  std::map<unsigned int, unsigned int> atomTypeToBasisInfoStartNum;

  unsigned int maxBasisShell = 0; // some lower unreachable value
  // unsigned int minBasisShell = 10; // some unreachable value

  unsigned int tmpBasisDim, b1, b2;

  // b1 and b2 are the range of principal quantum numbers, ends inclusive
  // where the slater type basis are constructed based on n, l, m values or each
  // orbital


  unsigned int basisHierarchyStart, basisHierarchyEnd, basisCount = 1;

  std::string basisInfoFile = "BasisInfo.inp";



  std::vector<AtomicOrbitalBasisManager> atomTypewiseSTOvector;
  // atomTypewiseSTOvector.reserve(numOfAtomTypes);
  int                           atomType;
  std::vector<std::vector<int>> atomTypesorbitals;
  readBasisFile(3, atomTypesorbitals, "BasisInfo.inp");


  for (unsigned int i = 0; i < numOfAtomTypes; ++i)
    {
      atomType = atomTypesVec[i];
      atomTypewiseSTOvector.push_back(AtomicOrbitalBasisManager(
        atomType, d_dftParamsPtr->AtomicOrbitalBasis, true));
    }
  /*** constructQuantumNumbersHierarchy */
  std::vector<std::vector<int>> atomTypewiseOrbitalist;
  std::vector<bool>             atomTypeflag(numOfAtomTypes, false);
  std::vector<int>              atomTypeoritalstart(numOfAtomTypes, 0);
  int                           counter = 1;
  for (int i = 0; i < atomTypesorbitals.size(); i++)
    {
      for (int j = 0; j < atomTypewiseSTOvector.size(); j++)
        {
          if (atomTypewiseSTOvector[j].atomType == atomTypesorbitals[i][0])
            {
              if (atomTypeflag[j] == false)
                {
                  atomTypeoritalstart[j] = counter;
                  atomTypeflag[j]        = true;
                }
              int n, l, m;
              n = atomTypesorbitals[i][1];
              l = atomTypesorbitals[i][2];

              for (m = -l; m <= l; m++)
                {
                  std::vector<int> tempvec(3, 0);
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
      writeOrbitalDataIntoFile(atomTypewiseOrbitalist,
                               "atomTypeWiseOrbitalNums.txt");
    }

  for (int j = 0; j < atomTypewiseSTOvector.size(); j++)
    {
      atomTypewiseSTOvector[j].CreatePseudoAtomicOrbitalBasis();
    }

  pcout << "vector of objects constructed!\n";



  std::ofstream atomWiseAtomicOrbitalInfoFile("atomWiseAtomicOrbitalInfo.txt");

  if (!atomWiseAtomicOrbitalInfoFile.is_open())
    {
      std::cerr << "Couldn't open "
                << "atomWiseAtomicOrbitalInfo.txt"
                << " file!!" << std::endl;
      exit(0);
    }

  unsigned int atomicNum;

  count = 0;
  std::vector<unsigned int> atomwiseGlobalbasisNum; // cumulative vector
  atomwiseGlobalbasisNum.reserve(numOfAtoms + 1);

  atomwiseGlobalbasisNum.push_back(count); // first entry is zero


  /* Change the above loop to: */
  for (int i = 0; i < atomicNumVec.size(); i++)
    {
      for (int j = 0; j < atomTypewiseSTOvector.size(); j++)
        {
          if (atomicNumVec[i] == atomTypewiseSTOvector[j].atomType)
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

  for (unsigned int i = 0; i < numOfAtoms; ++i)
    {
      atomicNum = atomicNumVec[i];
      tmp1      = atomTypetoAtomTypeID[atomicNum];
      tmp2      = atomwiseGlobalbasisNum[i];
      tmp3      = atomwiseGlobalbasisNum[i + 1];
      // nstart = atomTypetoNstart[ atomicNum ];
      // basisNstart = numOfOrbitalsForShellCount(1, nstart - 1);

      atomWiseAtomicOrbitalInfoFile << atomicNum << " " << tmp2 + 1 << " "
                                    << tmp3 << " " << atomTypeoritalstart[tmp1]
                                    << '\n';

      for (unsigned int j = tmp2; j < tmp3; ++j)
        {
          // globalBasisInfo.push_back({i, tmp1, basisNstart + j - tmp2});
          // globalBasisInfo.push_back({i, tmp1,atomTypewiseSTOvector[tmp1].n
          // ,atomTypewiseSTOvector[tmp1].l ,atomTypewiseSTOvector[tmp1].m});
          LocalAtomicBasisInfo temp;
          temp.atomID     = i;
          temp.atomTypeID = tmp1;
          temp.n          = atomTypewiseSTOvector[tmp1].n[j - tmp2];
          temp.l          = atomTypewiseSTOvector[tmp1].l[j - tmp2];
          temp.m          = atomTypewiseSTOvector[tmp1].m[j - tmp2];
          globalBasisInfo.push_back(temp);
          //  globalBasisInfo.push_back({i, tmp1,atomTypewiseSTOvector[tmp1].n[j
          //  - tmp2] ,atomTypewiseSTOvector[tmp1].l[j - tmp2]
          //  ,atomTypewiseSTOvector[tmp1].m[j - tmp2]});
          // globalBasisInfo.push_back({i, tmp1,atomTypewiseSTOvector[tmp1].n[j
          // - tmp2] ,atomTypewiseSTOvector[tmp1].l[j - tmp2]
          // ,atomTypewiseSTOvector[tmp1].m[j - tmp2]});
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



  occupationNum.resize(numOfKSOrbitals *
                       (d_dftParamsPtr->spinPolarized == 1 ? 2 : 1));
  pcout << "Size of Occupance vector" << occupationNum.size() << std::endl;

  unsigned int numEigenValues =
    eigenValuesInput[0].size() / (1 + d_dftParamsPtr->spinPolarized);
  pcout << "Number of Eigenvalues " << numEigenValues << std::endl;
  for (unsigned int iEigen = 0; iEigen < numOfKSOrbitals; ++iEigen)
    {
      if (d_dftParamsPtr->spinPolarized == 0)
        {
          occupationNum[iEigen] =
            dftUtils::getPartialOccupancy(eigenValuesInput[0][iEigen],
                                          fermiEnergy,
                                          C_kb,
                                          d_dftParamsPtr->TVal);
          // pcout<<occupationNum[iEigen]<<std::endl;
        }
      else
        {
          occupationNum[iEigen] =
            dftUtils::getPartialOccupancy(eigenValuesInput[0][iEigen],
                                          fermiEnergy,
                                          C_kb,
                                          d_dftParamsPtr->TVal);
          occupationNum[iEigen + numOfKSOrbitals] =
            dftUtils::getPartialOccupancy(
              eigenValuesInput[0][iEigen + numEigenValues],
              fermiEnergy,
              C_kb,
              d_dftParamsPtr->TVal);
        }
    }


  // Loop over atomic orbitals to evaluate at all nodal points

  const IndexSet &locallyOwnedSet = dofHandlerEigen.locally_owned_dofs();
  std::vector<IndexSet::size_type> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  unsigned int n_dofs = locallyOwnedDOFs.size();
  MPI_Barrier(MPI_COMM_WORLD);
  // std::cout<<"Processor ID: "<<this_mpi_process<<" has dofs total:
  // "<<n_dofs<<std::endl;
  std::vector<double> scaledOrbitalValues_FEnodes(n_dofs * totalDimOfBasis,
                                                  0.0);
  std::vector<double> scaledKSOrbitalValues_FEnodes(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 0 : 1), 0.0);
  std::vector<double> scaledKSOrbitalValues_FEnodes_spinup(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 1 : 0), 0.0);
  std::vector<double> scaledKSOrbitalValues_FEnodes_spindown(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 1 : 0), 0.0);
  if (this_mpi_process == 0)
    {
      // and writing the high level basis information

      std::ofstream highLevelBasisInfoFile("highLevelBasisInfo.txt");

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
  double timerCreatingMatrices = MPI_Wtime();
  double r,theta,phi;
  int SumCounter=0;

#ifdef USE_COMPLEX

#else


  for (unsigned int dof = 0; dof < n_dofs; ++dof)
    {
      // get nodeID
      const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
      if (!constraintsNone.is_constrained(dofID))
        {
          // get coordinates of the finite-element node
          Point<3> node = d_supportPointsEigen[dofID];

          auto count1 = totalDimOfBasis * dof;

          for (unsigned int i = 0; i < totalDimOfBasis; ++i)
            {
              auto atomTypeID   = globalBasisInfo[i].atomTypeID;
              auto atomChargeID = globalBasisInfo[i].atomID;

              std::vector<int> imageIdsList;
              if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
                  d_dftParamsPtr->periodicZ)
                
                {
                  imageIdsList = d_globalChargeIdToImageIdMap[atomChargeID];
                }
              else
                {
                  imageIdsList.push_back(atomChargeID);
                }
 
                   OrbitalQuantumNumbers orbital = {globalBasisInfo[i].n,
                                                   globalBasisInfo[i].l,
                                                   globalBasisInfo[i].m};
              //pcout<<"imageIdsList.size(): "<<imageIdsList.size()<<std::endl;
              int counterlist = 0;
              for (int imageID = 0; imageID < imageIdsList.size(); imageID++)
                {
                  int                 chargeId = imageIdsList[imageID];
                  std::vector<double> atomPos(3, 0.0);
                  if (chargeId < numOfAtoms)
                    {
                      atomPos[0] = atomLocations[chargeId][2];
                      atomPos[1] = atomLocations[chargeId][3];
                      atomPos[2] = atomLocations[chargeId][4];
                    }
                  else
                    {
                      atomPos[0] = d_imagePositions[chargeId - numOfAtoms][0];
                      atomPos[1] = d_imagePositions[chargeId - numOfAtoms][1];
                      atomPos[2] = d_imagePositions[chargeId - numOfAtoms][2];
                    }
                    auto relativeEvalPoint = relativeVector3d(node, atomPos);

                    convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);
                 
                  if(atomTypewiseSTOvector[atomTypeID].maxRadialcutoff < 0 || r <= atomTypewiseSTOvector[atomTypeID].maxRadialcutoff)
                  {
                    counterlist++;
                  if (d_dftParamsPtr->AtomicOrbitalBasis == 1)
                    {
                      scaledOrbitalValues_FEnodes[count1 + i] +=
                        d_kohnShamDFTOperatorPtr->d_sqrtMassVector
                          .local_element(dof) *
                        atomTypewiseSTOvector[atomTypeID].bungeOrbital(orbital,
                                                                       node,
                                                                       atomPos);
                    }
                  if (d_dftParamsPtr->AtomicOrbitalBasis == 0)
                    {
                      scaledOrbitalValues_FEnodes[count1 + i] +=
                        d_kohnShamDFTOperatorPtr->d_sqrtMassVector
                          .local_element(dof) *
                        atomTypewiseSTOvector[atomTypeID]
                          .PseudoAtomicOrbitalvalue(orbital, node, atomPos,r,theta,phi);
                    }
                  }  
                  
                }
                SumCounter+=counterlist;
            }
        }
      auto count2 = numOfKSOrbitals * dof;

      for (unsigned int j = 0; j < numOfKSOrbitals; ++j)
        {
          if (d_dftParamsPtr->spinPolarized == 1)
            {
              scaledKSOrbitalValues_FEnodes_spinup[count2 + j] =
                d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                d_eigenVectorsFlattenedSTL[0][dof * d_numEigenValues + j];
              // pcout<<"Accessing spin down"<<std::endl;
              scaledKSOrbitalValues_FEnodes_spindown[count2 + j] =
                d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                d_eigenVectorsFlattenedSTL[1][dof * d_numEigenValues + j];
            }
          else
            {
              scaledKSOrbitalValues_FEnodes[count2 + j] =
                d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                d_eigenVectorsFlattenedSTL[0][dof * d_numEigenValues + j];
            }
        }
    }


#endif
  MPI_Barrier(MPI_COMM_WORLD);
  timerCreatingMatrices = MPI_Wtime() - timerCreatingMatrices;
  pcout<<" Creating PHI and PSI matrices: "<<timerCreatingMatrices<<std::endl;
    MPI_Allreduce(&SumCounter,
                &SumCounter,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);  
  pcout<<"Sum of Counter: "<<SumCounter;
  MPI_Barrier(MPI_COMM_WORLD);
  double timerScompute = MPI_Wtime();  
  auto upperTriaOfSserial =
    selfMatrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis);
  std::vector<double> upperTriaOfS((totalDimOfBasis * (totalDimOfBasis + 1) /
                                    2),
                                   0.0);
  MPI_Allreduce(&upperTriaOfSserial[0],
                &upperTriaOfS[0],
                (totalDimOfBasis * (totalDimOfBasis + 1) / 2),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  timerScompute = MPI_Wtime() - timerScompute;
  pcout<<" Computing S matrix: "<<timerScompute<<std::endl;
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      writeVectorToFile(upperTriaOfS, "overlapMatrix.txt");
      // printVector(upperTriaOfS);
    }
   std::vector<double> S(totalDimOfBasis*totalDimOfBasis,0.0);
   int Scount=0;
   for(int i = 0; i < totalDimOfBasis; i++)
   {
    for (int j=0; j < totalDimOfBasis; j++)
    {
      if(j >=i)
      {
        S[i*totalDimOfBasis+j] = upperTriaOfS[Scount];
        S[j*totalDimOfBasis +i] = S[i*totalDimOfBasis+j];
        Scount++;
      }
    }
   }
  std::vector<double> D(totalDimOfBasis,0.0);
  std::vector<double> U(totalDimOfBasis*totalDimOfBasis,0.0);
  MPI_Barrier(MPI_COMM_WORLD);
  double timerSdiagonalization = MPI_Wtime();
  if(this_mpi_process == 0)
    U = diagonalization(S,totalDimOfBasis,D);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSdiagonalization = MPI_Wtime() - timerSdiagonalization;
  pcout<<"Diagonalization of S: "<<timerSdiagonalization<<std::endl;
    MPI_Bcast(
      &(D[0]), totalDimOfBasis, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    MPI_Bcast(
      &(U[0]), totalDimOfBasis*totalDimOfBasis, MPI_DOUBLE, 0, MPI_COMM_WORLD);       


  MPI_Barrier(MPI_COMM_WORLD);
  double timerUStranspose = MPI_Wtime();
  auto Ut = TransposeMatrix(U,totalDimOfBasis);
   MPI_Barrier(MPI_COMM_WORLD);
   timerUStranspose = MPI_Wtime()-timerUStranspose;


  MPI_Barrier(MPI_COMM_WORLD);
  double timerSinverse = MPI_Wtime();   
  auto invS = powerOfMatrix(-1,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSinverse = MPI_Wtime()-timerSinverse;
  pcout<<" Computing S^-1: "<<timerSinverse<<std::endl;




  MPI_Barrier(MPI_COMM_WORLD);
  double timerSminushalf = MPI_Wtime();
  auto Sminushalf = powerOfMatrix(-0.5,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSminushalf = MPI_Wtime() - timerSminushalf;
  

  MPI_Barrier(MPI_COMM_WORLD);
  double timerShalf = MPI_Wtime();
  auto Shalf = powerOfMatrix(0.5,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerShalf = MPI_Wtime() - timerShalf;  
  pcout<<" Computing S^0.5: "<<timerShalf<<std::endl;

      MPI_Barrier(MPI_COMM_WORLD);

      const unsigned int N = totalDimOfBasis;


double timerHprojnormal,timerSinvHproj,timerHprojScalapack,timerChat2,timerCbar2;

#ifdef USE_COMPLEX

#else
      //std::vector<dataTypes::number> CoeffNew(N * N, 0.0);
       
      std::vector<dataTypes::number> ProjHam1;

      MPI_Barrier(MPI_COMM_WORLD);
      timerHprojnormal = MPI_Wtime();  
      d_kohnShamDFTOperatorPtr->XtHX(scaledOrbitalValues_FEnodes,
                                     totalDimOfBasis,
                                     ProjHam1);
      MPI_Barrier(MPI_COMM_WORLD);
      timerHprojnormal = MPI_Wtime()-timerHprojnormal;
       pcout<<" Computing H projected: "<<timerHprojnormal<<std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      timerSinvHproj = MPI_Wtime();
      auto ProjHam2 = matrixmatrixmul(Sminushalf,N,N,ProjHam1,N,N);
      auto ProjHam = matrixmatrixmul(ProjHam2,N,N,Sminushalf,N,N);
      MPI_Barrier(MPI_COMM_WORLD);
      timerSinvHproj = MPI_Wtime()-timerSinvHproj;
      pcout<<" Compute Sinv Hproj: "<<timerSinvHproj<<std::endl;
      const unsigned int rowsBlockSize = d_elpaScala->getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        d_elpaScala->getProcessGridDftfeScalaWrapper();
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPar(totalDimOfBasis,
                                                           processGrid,
                                                           rowsBlockSize);
      MPI_Barrier(MPI_COMM_WORLD);
      timerHprojScalapack = MPI_Wtime();
      d_kohnShamDFTOperatorPtr->XtHX(scaledOrbitalValues_FEnodes,
                                     totalDimOfBasis,
                                     processGrid,
                                     projHamPar);
      MPI_Barrier(MPI_COMM_WORLD);
      timerHprojScalapack =MPI_Wtime()-timerHprojScalapack;
      pcout<<" Compute H projected SCALAPACK: "<<timerHprojScalapack<<std::endl;

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          writeVectorAs2DMatrix(ProjHam,
                                totalDimOfBasis,
                                totalDimOfBasis,
                                "ProjectedHamilton.txt");
          // printVector(ProjHam);
          pcout << std::endl;
        }
      
	std::vector<double> projEnergy(N,0.0);
	std::vector<double> CoeffNew(N*N,0.0);
	MPI_Barrier(MPI_COMM_WORLD);
  timerChat2 = MPI_Wtime();
       if(this_mpi_process == 0)
        CoeffNew = diagonalization(ProjHam,N,projEnergy);
        MPI_Barrier(MPI_COMM_WORLD);
      timerChat2 = MPI_Wtime() - timerChat2;    
      pcout<<" Computing Chat (Diagonlaization of Hp): "<<timerChat2<<std::endl;
     MPI_Bcast(
      &(projEnergy[0]), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(
      &(CoeffNew[0]), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);     
	//MPI_Barrier(MPI_COMM_WORLD);
    //  timerChat2 = MPI_Wtime() - timerChat2;
      if (this_mpi_process == 0)
        writeVectorAs2DMatrix(CoeffNew,
                              totalDimOfBasis,
                              totalDimOfBasis,
                              "FePHP_v2.txt");
      
     
      MPI_Barrier(MPI_COMM_WORLD);
      timerCbar2 = MPI_Wtime();
      auto C_bar2 = matrixmatrixTmul(Sminushalf,
                                     totalDimOfBasis,
                                     totalDimOfBasis,
                                     CoeffNew,
                                     totalDimOfBasis,
                                     numOfKSOrbitals);
      MPI_Barrier(MPI_COMM_WORLD);
      timerCbar2 = MPI_Wtime()- timerCbar2;     
      pcout<<" Computing Cbar: "<<timerCbar2<<std::endl;
      if (this_mpi_process == 0)
        {
          writeVectorAs2DMatrix(C_bar2,
                                totalDimOfBasis,
                                numOfKSOrbitals,
                                "FePOP_v2.txt");
          pcout << std::endl;
        }
#endif


  pcout<<std::endl;
   pcout<<"----------------------------------------------------------"<<std::endl;
  pcout<<"------------------- NEW METHOD ----------------------------"<<std::endl;
  pcout<<" Creating PHI and PSI matrices: "<<timerCreatingMatrices<<std::endl;
  pcout<<" Computing S matrix: "<<timerScompute<<std::endl;
  pcout<<"Diagonalization of S: "<<timerSdiagonalization<<std::endl; 
  pcout<<"Computing S^-0.5: "<<timerSminushalf<<std::endl;
  pcout<<" Computing H projected: "<<timerHprojnormal<<std::endl;
  pcout<<" Compute H projected SCALAPACK: "<<timerHprojScalapack<<std::endl;
  pcout<<" Compute Sinv Hproj: "<<timerSinvHproj<<std::endl;
  pcout<<" Computing Chat (Diagonlaization of Hp): "<<timerChat2<<std::endl;
  pcout<<" Computing Cbar: "<<timerCbar2<<std::endl;
  pcout<<" Compute U transpose: "<<timerUStranspose<<std::endl ;
  pcout<<" TOTAL TIME in sec: "<<timerScompute+timerSdiagonalization+timerHprojnormal+timerSinvHproj+
  timerCbar2+timerSminushalf+timerChat2<<std::endl;
  pcout<<" TOTAL TIME in sec: "<<timerScompute+timerSdiagonalization+timerHprojScalapack+timerSinvHproj+
  timerCbar2+timerSminushalf+timerChat2<<std::endl;

	


  
    
}
