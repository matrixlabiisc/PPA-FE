void
writeOrbitalDataIntoFile(const std::vector<std::vector<int>> &data,
                         const std::string &                  fileName)
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
                  outFile << data[irow][icol];
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
readBasisFile(const unsigned int             numColumns,
              std::vector<std::vector<int>> &data,
              const std::string &            fileName)
{
  std::vector<int> rowData(numColumns, 0.0);
  std::ifstream    readFile(fileName.c_str());
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
// // corresponding atomPositions should be as an external array or suitable in
// a datastructure

void
constructQuantumNumbersHierarchy(unsigned int      n,
                                 unsigned int      l,
                                 std::vector<int> &rank)
{
  // assume the vector of size 0 has already been reserved with space for N
  // shells which is N(N+1)(2N+1)/6 orbitals for N shells N is maximum of the
  // principal quantum number over each atomType this function is called just
  // once for the whole program

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

void
appendElemsOfRangeToFile(unsigned int start,
                         unsigned int end,
                         std::string  filename)
{
  std::ofstream outputFile;
  outputFile.open(filename, std::ofstream::out | std::ofstream::app);

  if (outputFile.is_open())
    {
      for (int i = start; i <= end; ++i)
        {
          outputFile << i << '\n';
        }
    }

  else
    {
      std::cerr << "Couldn't open " << filename << " file!!" << std::endl;
      exit(0);
    }

  outputFile.close();
  // it is usually not required to close the file
}
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::newRhoSpillFactor(
  const dealii::DoFHandler<3> &                        dofHandlerOfField,
  const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues,
  const std::map<dealii::CellId, std::vector<double>> *NewrhoQuadValues)
{
  double               Numerator   = 0.0;
  double               Denominator = 0.0;
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
              Denominator += (rhoValues[q_point] * fe_values.JxW(q_point)) *
                             (rhoValues[q_point]);
              Numerator += (rhoValues[q_point] - NewrhoValues[q_point]) *
                           (rhoValues[q_point] - NewrhoValues[q_point]) *
                           fe_values.JxW(q_point);
            }
        }
    }
  return (sqrt(Utilities::MPI::sum(Numerator, mpi_communicator)) /
          sqrt(Utilities::MPI::sum(Denominator, mpi_communicator)));
}
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::newRhoSpillFactorspin(
  const dealii::DoFHandler<3> &                        dofHandlerOfField,
  const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues,
  const std::map<dealii::CellId, std::vector<double>> *NewrhoQuadValues,
  int                                                  spinIndex)
{
  double               Numerator   = 0.0;
  double               Denominator = 0.0;
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
              Denominator +=
                (rhoValues[2 * q_point + spinIndex] * fe_values.JxW(q_point)) *
                (rhoValues[2 * q_point + spinIndex]);
              Numerator += (rhoValues[2 * q_point + spinIndex] -
                            NewrhoValues[2 * q_point + spinIndex]) *
                           (rhoValues[2 * q_point + spinIndex] -
                            NewrhoValues[2 * q_point + spinIndex]) *
                           fe_values.JxW(q_point);
            }
        }
    }
  return (sqrt(Utilities::MPI::sum(Numerator, mpi_communicator)) /
          sqrt(Utilities::MPI::sum(Denominator, mpi_communicator)));
}
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::orbitalPopulationCompute(
  const std::vector<std::vector<double>> &eigenValuesInput, unsigned int kpoint)
{
  
  double timerUOtranspose,timerUStranspose,timerHprojOrbital,timerChatcompute,
    timerShalf,timerCbarcompute,timerOminushalf,timerOdiagnolaization,timerOcompute,
    timerCcompute,timerSinverse,timerSdiagonalization,timerPhiTPsi,timerScompute;
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
    eigenValuesInput[kpoint].size() / (1 + d_dftParamsPtr->spinPolarized);
  pcout << "Number of Eigenvalues " << numEigenValues << std::endl;
  for (unsigned int iEigen = 0; iEigen < numOfKSOrbitals; ++iEigen)
    {
      if (d_dftParamsPtr->spinPolarized == 0)
        {
          occupationNum[iEigen] =
            dftUtils::getPartialOccupancy(eigenValuesInput[kpoint][iEigen],
                                          fermiEnergy,
                                          C_kb,
                                          d_dftParamsPtr->TVal);
          // pcout<<occupationNum[iEigen]<<std::endl;
        }
      else
        {
          occupationNum[iEigen] =
            dftUtils::getPartialOccupancy(eigenValuesInput[kpoint][iEigen],
                                          fermiEnergy,
                                          C_kb,
                                          d_dftParamsPtr->TVal);
          occupationNum[iEigen + numOfKSOrbitals] =
            dftUtils::getPartialOccupancy(
              eigenValuesInput[kpoint][iEigen + numEigenValues],
              fermiEnergy,
              C_kb,
              d_dftParamsPtr->TVal);
        }
    }


  // Loop over atomic orbitals to evaluate at all nodal points

  const IndexSet &locallyOwnedSet = dofHandler.locally_owned_dofs();
  std::vector<IndexSet::size_type> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  unsigned int n_dofs = locallyOwnedDOFs.size();
  pcout<<"Total DOFs: "<<n_dofs<<std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  // std::cout<<"Processor ID: "<<this_mpi_process<<" has dofs total:
  // "<<n_dofs<<std::endl;
#ifdef USE_COMPLEX
  
  const std::complex<double> iota(0, 1);
  pcout<<"Imaginary No: "<<iota<<std::endl;
  std::vector<std::complex<double>> scaledOrbitalValues_FEnodes(n_dofs * totalDimOfBasis,
                                                  std::complex<double> (0,0));
  std::vector<std::complex<double>> scaledKSOrbitalValues_FEnodes(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 0 : 1), std::complex<double> (0,0));
  std::vector<std::complex<double>> scaledKSOrbitalValues_FEnodes_spinup(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 1 : 0), std::complex<double> (0,0));
  std::vector<std::complex<double>> scaledKSOrbitalValues_FEnodes_spindown(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 1 : 0), std::complex<double> (0,0));
#else


  std::vector<double> scaledOrbitalValues_FEnodes(n_dofs * totalDimOfBasis,
                                                  0.0);
  std::vector<double> scaledKSOrbitalValues_FEnodes(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 0 : 1), 0.0);
  std::vector<double> scaledKSOrbitalValues_FEnodes_spinup(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 1 : 0), 0.0);
  std::vector<double> scaledKSOrbitalValues_FEnodes_spindown(
    (n_dofs * numOfKSOrbitals) * (d_dftParamsPtr->spinPolarized ? 1 : 0), 0.0);
  
#endif  
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
pcout<<"K-point coordinate: "<<d_kPointCoordinates[kpoint*3+0]<<" "<<d_kPointCoordinates[kpoint*3+1]<<" "<<d_kPointCoordinates[kpoint*3+2]<<std::endl;
  for (unsigned int dof = 0; dof < n_dofs; ++dof)
    {
      //pcout<<"Node no: "<<dof<<std::endl;
      // get nodeID
      const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
          
          double kdotx,kdotRm; 
         //pcout<<"kdotx at DOF:" <<dofID<<" :"<<kdotx<<" "<<exp(iota*kdotx)<<std::endl;               

      if (!constraintsNone.is_constrained(dofID))
        {
          // get coordinates of the finite-element node
          Point<3> node = d_supportPoints[dofID];


          kdotx = d_kPointCoordinates[kpoint*3+0]*node[0]+
                        d_kPointCoordinates[kpoint*3+1]*node[1] +
                         d_kPointCoordinates[kpoint*3+2]*node[2];
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
                    //pcout<<"kdotRm at imageId:" <<imageID<<" :"<<kdotRm<<" "<<exp(iota*kdotRm)<<std::endl;
                    counterlist++;
                   kdotRm = atomPos[0]*d_kPointCoordinates[kpoint*3+0] +
                                                 atomPos[1]*d_kPointCoordinates[kpoint*3+1] +
                                                  atomPos[2]*d_kPointCoordinates[kpoint*3+2] ;
                  if (d_dftParamsPtr->AtomicOrbitalBasis == 1)
                    {
                      double realvalue = scaledOrbitalValues_FEnodes[count1 + i].real();
                      double imaginaryValue = scaledOrbitalValues_FEnodes[count1 + i].imag();

                      scaledOrbitalValues_FEnodes[count1 + i] +=
                        d_kohnShamDFTOperatorPtr->d_sqrtMassVector
                          .local_element(dof) *
                        atomTypewiseSTOvector[atomTypeID].bungeOrbital(orbital,
                                                                       node,
                                                                       atomPos);//*std::exp(iota*kdotRm);
                    }
                  if (d_dftParamsPtr->AtomicOrbitalBasis == 0)
                    {
                      double realvalue = scaledOrbitalValues_FEnodes[count1 + i].real();
                      double imaginaryValue = scaledOrbitalValues_FEnodes[count1 + i].imag();
                      realvalue += d_kohnShamDFTOperatorPtr->d_sqrtMassVector
                          .local_element(dof) *
                        atomTypewiseSTOvector[atomTypeID]
                          .PseudoAtomicOrbitalvalue(orbital, node, atomPos,r,theta,phi)*std::cos(kdotRm);
                      imaginaryValue += d_kohnShamDFTOperatorPtr->d_sqrtMassVector
                          .local_element(dof) *
                        atomTypewiseSTOvector[atomTypeID]
                          .PseudoAtomicOrbitalvalue(orbital, node, atomPos,r,theta,phi)*std::sin(kdotRm);                      
                      
                      scaledOrbitalValues_FEnodes[count1 + i].real(realvalue);
                      scaledOrbitalValues_FEnodes[count1 + i].imag(imaginaryValue);
                        
                    }
                  }  
                  
                }
                SumCounter+=counterlist;
            }
        
      //pcout<<" Line No 699"<<" "<<d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof)<<std::endl;  
      auto count2 = numOfKSOrbitals * dof;

      for (unsigned int j = 0; j < numOfKSOrbitals; ++j)
        {
          if (d_dftParamsPtr->spinPolarized == 1)
            {
              scaledKSOrbitalValues_FEnodes_spinup[count2 + j] =
                d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                d_eigenVectorsFlattenedSTL[2*kpoint+0][dof * d_numEigenValues + j]*std::exp(iota*kdotx);
              // pcout<<"Accessing spin down"<<std::endl;
              scaledKSOrbitalValues_FEnodes_spindown[count2 + j] =
                d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                d_eigenVectorsFlattenedSTL[2*kpoint +1][dof * d_numEigenValues + j]*std::exp(iota*kdotx);
            }
          else
            {
              //pcout<<"  Line 712"<<std::endl;
              
              scaledKSOrbitalValues_FEnodes[count2 + j] += d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                (d_eigenVectorsFlattenedSTL[kpoint][dof * d_numEigenValues + j]*std::exp(iota*kdotx));
              /*double realValue = scaledKSOrbitalValues_FEnodes[count2 + j].real();
              double imaginaryValue =  scaledKSOrbitalValues_FEnodes[count2 + j].imag();
                realValue += d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                d_eigenVectorsFlattenedSTL[kpoint][dof * d_numEigenValues + j]*std::cos(kdotx);
                imaginaryValue += d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof) *
                d_eigenVectorsFlattenedSTL[kpoint][dof * d_numEigenValues + j]*std::sin(kdotx);
                scaledKSOrbitalValues_FEnodes[count2 + j].real(realValue);
                scaledKSOrbitalValues_FEnodes[count2 + j].imag(imaginaryValue); */
            }
        }
        }
        //pcout<<" Line 722"<<std::endl;
    }
  //pcout<<"Line no 718"<<std::endl;  
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
   timerScompute = MPI_Wtime();  
  auto upperTriaOfSserial =
    selfMatrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis);
  std::vector<std::complex<double>> upperTriaOfS((totalDimOfBasis * (totalDimOfBasis + 1) /
                                    2),
                                   0.0);
  MPI_Allreduce(&upperTriaOfSserial[0],
                &upperTriaOfS[0],
                totalDimOfBasis * (totalDimOfBasis + 1) / 2,
                dataTypes::mpi_type_id(&upperTriaOfSserial[0]),
                MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  timerScompute = MPI_Wtime() - timerScompute;
  pcout<<" Computing S matrix: "<<timerScompute<<std::endl;
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      writeVectorToFile(upperTriaOfS, "overlapMatrixComplex.txt");
      // printVector(upperTriaOfS);
    }
   std::vector<std::complex<double>> S(totalDimOfBasis*totalDimOfBasis,std::complex<double> (0,0));
   pcout<<"Printing the S overlap matrix: "<<std::endl;
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
      pcout<<S[j*totalDimOfBasis +i]<<" ";
    }
      pcout<<std::endl;
   }
  std::vector<double> D(totalDimOfBasis,0.0);
  std::vector<std::complex<double>> U(totalDimOfBasis*totalDimOfBasis,std::complex<double> (0,0));
  MPI_Barrier(MPI_COMM_WORLD);
  timerSdiagonalization = MPI_Wtime();
  if(this_mpi_process == 0)
    U = diagonalization(S,totalDimOfBasis,D);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSdiagonalization = MPI_Wtime() - timerSdiagonalization;
  pcout<<"Diagonalization of S: "<<timerSdiagonalization<<std::endl;
    MPI_Bcast(
      &(D[0]), totalDimOfBasis, dataTypes::mpi_type_id(&D[0]), 0, MPI_COMM_WORLD); 
    MPI_Bcast(
      &(U[0]), totalDimOfBasis*totalDimOfBasis, dataTypes::mpi_type_id(&U[0]), 0, MPI_COMM_WORLD);       
  pcout<<"Eigenvalues of S: "<<std::endl;
  for (int i = 0; i < totalDimOfBasis; i++)
    pcout<<D[i]<<" ";
  pcout<<std::endl;
  pcout<<"Eigenvectors of S: "<<std::endl;
  for(int i = 0;i < totalDimOfBasis; i++)
    {
      for(int j = 0; j < totalDimOfBasis; j++)
        pcout<<U[i*totalDimOfBasis+j]<<" ";
      pcout<<std::endl;  
    }  
  pcout<<"-------------------------"<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);


   timerUStranspose = MPI_Wtime();
  auto Ut = TransposeMatrix(U,totalDimOfBasis);
   MPI_Barrier(MPI_COMM_WORLD);
   timerUStranspose = MPI_Wtime()-timerUStranspose;


  MPI_Barrier(MPI_COMM_WORLD);
   timerSinverse = MPI_Wtime();   
  auto invS = powerOfMatrix(-1,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSinverse = MPI_Wtime()-timerSinverse;
  pcout<<" Computing S^-1: "<<timerSinverse<<std::endl;
  pcout<<"Elements of invserse S: "<<std::endl;
  for(int i = 0;i < totalDimOfBasis; i++)
    {
      for(int j = 0; j < totalDimOfBasis; j++)
        pcout<<invS[i*totalDimOfBasis+j]<<" ";
      pcout<<std::endl;  
    }  
  pcout<<"-------------------------"<<std::endl;



  MPI_Barrier(MPI_COMM_WORLD);
  double timerSminushalf = MPI_Wtime();
  auto Sminushalf = powerOfMatrix(-0.5,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSminushalf = MPI_Wtime() - timerSminushalf;
  

  MPI_Barrier(MPI_COMM_WORLD);
   timerShalf = MPI_Wtime();
  auto Shalf = powerOfMatrix(0.5,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerShalf = MPI_Wtime() - timerShalf;  
  pcout<<" Computing S^0.5: "<<timerShalf<<std::endl;
  pcout<<"Elements of  S^0.5: "<<std::endl;
  for(int i = 0;i < totalDimOfBasis; i++)
    {
      for(int j = 0; j < totalDimOfBasis; j++)
        pcout<<Shalf[i*totalDimOfBasis+j]<<" ";
      pcout<<std::endl;  
    }  
  pcout<<"-------------------------"<<std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
       timerPhiTPsi = MPI_Wtime();
      auto arrayVecOfProjserial =
        matrixTmatrixmul(scaledOrbitalValues_FEnodes,
                         n_dofs,
                         totalDimOfBasis,
                         scaledKSOrbitalValues_FEnodes,
                         n_dofs,
                         numOfKSOrbitals);
      std::vector<std::complex<double>> arrayVecOfProj(totalDimOfBasis * numOfKSOrbitals,
                                         0.0);
      MPI_Allreduce(&arrayVecOfProjserial[0],
                    &arrayVecOfProj[0],
                    (totalDimOfBasis * numOfKSOrbitals),
                     dataTypes::mpi_type_id(&arrayVecOfProjserial[0]),
                    MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      timerPhiTPsi = MPI_Wtime() - timerPhiTPsi;
      pcout<<"Computing PHI^TPSI: "<<timerPhiTPsi<<std::endl;
      pcout << "Matrix of projections with atomic orbitals: \n";
      MPI_Barrier(MPI_COMM_WORLD);
       timerCcompute = MPI_Wtime();
      auto coeffArrayVecOfProj = matrixmatrixmul(invS,
                                                 totalDimOfBasis,
                                                 totalDimOfBasis,
                                                 arrayVecOfProj,
                                                 totalDimOfBasis,
                                                 numOfKSOrbitals);
      MPI_Barrier(MPI_COMM_WORLD);
      timerCcompute = MPI_Wtime() - timerCcompute;  
      pcout<<"Computing C: "<<timerCcompute<<std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
       timerOcompute = MPI_Wtime();
      auto B = matrixTmatrixmul(coeffArrayVecOfProj,totalDimOfBasis,numOfKSOrbitals,S,totalDimOfBasis,totalDimOfBasis);
      auto O = matrixmatrixmul(B,numOfKSOrbitals,totalDimOfBasis,coeffArrayVecOfProj,totalDimOfBasis,numOfKSOrbitals);                                               
      MPI_Barrier(MPI_COMM_WORLD);
      timerOcompute = MPI_Wtime() -timerOcompute;
      pcout<<" Computing O: "<<timerOcompute<<std::endl;
      std::vector<double> D_O(numOfKSOrbitals,0.0);
      std::vector<std::complex<double>> U_O(numOfKSOrbitals*numOfKSOrbitals,0.0);
      
      MPI_Barrier(MPI_COMM_WORLD);
       timerOdiagnolaization = MPI_Wtime();
       if(this_mpi_process == 0)
	U_O =  diagonalization(O,numOfKSOrbitals,D_O);
      MPI_Barrier(MPI_COMM_WORLD);
      timerOdiagnolaization = MPI_Wtime()-timerOdiagnolaization; 
    pcout<<" Diagonalization of O: "<<timerOdiagnolaization<<std::endl;     
    MPI_Bcast(
      &(D_O[0]), numOfKSOrbitals, dataTypes::mpi_type_id(&D_O[0]), 0, MPI_COMM_WORLD); 
    MPI_Bcast(
      &(U_O[0]), numOfKSOrbitals*numOfKSOrbitals, dataTypes::mpi_type_id(&U_O[0]), 0, MPI_COMM_WORLD); 
     // MPI_Barrier(MPI_COMM_WORLD);
     // timerOdiagnolaization = MPI_Wtime()-timerOdiagnolaization;

  MPI_Barrier(MPI_COMM_WORLD);
   timerUOtranspose = MPI_Wtime();
  auto U_Ot = TransposeMatrix(U_O,numOfKSOrbitals);
   MPI_Barrier(MPI_COMM_WORLD);
   timerUOtranspose = MPI_Wtime()-timerUOtranspose;  
  
  
  
      MPI_Barrier(MPI_COMM_WORLD);
       timerOminushalf = MPI_Wtime();
      auto Ominushalf = powerOfMatrix(-0.5,D_O,U_Ot,numOfKSOrbitals,U_Ot);
      MPI_Barrier(MPI_COMM_WORLD);
      timerOminushalf = MPI_Wtime() - timerOminushalf;
      pcout<<" Computing O^-0.5: "<<timerOminushalf<<std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
       timerCbarcompute = MPI_Wtime();
      std::vector<std::complex<double>> C_bar = matrixmatrixmul(coeffArrayVecOfProj,totalDimOfBasis,numOfKSOrbitals,
                                    Ominushalf,numOfKSOrbitals,numOfKSOrbitals);
      MPI_Barrier(MPI_COMM_WORLD);
      timerCbarcompute = MPI_Wtime() - timerCbarcompute;
      pcout<<" Computing Cbar: "<<timerCbarcompute<<std::endl;

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          writeVectorAs2DMatrix(C_bar,
                                totalDimOfBasis,
                                numOfKSOrbitals,
                                "FePOP_v1Complex.txt");
          // printVector(CoeffofOrthonormalisedKSonAO);
        }
      MPI_Barrier(MPI_COMM_WORLD);
       timerChatcompute = MPI_Wtime();
      std::vector<std::complex<double>> C_hat = matrixmatrixmul(Shalf,totalDimOfBasis,totalDimOfBasis,C_bar,totalDimOfBasis,numOfKSOrbitals);
      MPI_Barrier(MPI_COMM_WORLD);
      timerChatcompute = MPI_Wtime() - timerChatcompute; 
      pcout<<" Computing Chat: "<<timerChatcompute<<std::endl;
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          writeVectorAs2DMatrix(C_hat,
                                totalDimOfBasis,
                                numOfKSOrbitals,
                                "FePHP_v1Complex.txt");
        }

        MPI_Barrier(MPI_COMM_WORLD);
         timerHprojOrbital = MPI_Wtime();
        auto Hproj_orbital = computeHprojOrbital(C_hat,C_hat,
                                               totalDimOfBasis,
                                               numOfKSOrbitals,
                                               eigenValues[kpoint]);
        MPI_Barrier(MPI_COMM_WORLD);
        timerHprojOrbital = MPI_Wtime()-timerHprojOrbital;                                       
        pcout<<" Computing Projected Hamiltonian: "<<timerHprojOrbital<<std::endl;
      
      
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          writeVectorAs2DMatrix(Hproj_orbital,
                                totalDimOfBasis,
                                totalDimOfBasis,
                                "Hproj_orbitalCOmplex.txt");
        } 
      // Compute projected Hamiltonian of FE discretized Hamiltonian into
      
        Hproj_orbital.clear();
	C_hat.clear();
	C_bar.clear();	
	O.clear();
	U_O.clear();
	B.clear();
	
	std::vector<std::complex<double>>().swap(Hproj_orbital);
	std::vector<std::complex<double>>().swap(C_hat);
	std::vector<std::complex<double>>().swap(C_bar);
	std::vector<std::complex<double>>().swap(O);
	std::vector<std::complex<double>>().swap(U_O);
	std::vector<std::complex<double>>().swap(B);

	pcout
        << "--------------------------COHP Data Saved------------------------------"
        << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      if (this_mpi_process == 0)
        {
          // writing the energy levels and the occupation numbers
          unsigned int  kPointDummy = 0;
          std::ofstream energyLevelsOccNumsFile("energyLevelsOccNums.txt");

          if (energyLevelsOccNumsFile.is_open())
            {
              for (unsigned int i = 0; i < eigenValues[0].size(); ++i)
                {
                  const double partialOccupancy =
                    dftUtils::getPartialOccupancy(eigenValues[kPointDummy][i],
                                                  fermiEnergy,
                                                  C_kb,
                                                  d_dftParamsPtr->TVal);

                  energyLevelsOccNumsFile << eigenValues[kPointDummy][i] << " "
                                          << partialOccupancy << '\n';
                }

              energyLevelsOccNumsFile.close();
            }

          else
            pcout << "couldn't open energyLevelsOccNums.txt file!\n";
        }
      if (this_mpi_process == 0)
        {
          pcout
            << "\n-------------------------------------------------------\n";
          pcout << "Projected SpillFactors are:" << std::endl;
          spillFactorsofProjectionwithCS(coeffArrayVecOfProj,
                                         upperTriaOfS,
                                         occupationNum,
                                         totalDimOfBasis,
                                         numOfKSOrbitals,
                                         totalDimOfBasis,
                                         totalDimOfBasis);
          pcout
            << "\n-------------------------------------------------------\n";
        }




#else


  for (unsigned int dof = 0; dof < n_dofs; ++dof)
    {
      // get nodeID
      const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
      if (!constraintsNone.is_constrained(dofID))
        {
          // get coordinates of the finite-element node
          Point<3> node = d_supportPoints[dofID];

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
        
      //pcout<<" Line No 1099"<<" "<<d_kohnShamDFTOperatorPtr->d_sqrtMassVector.local_element(dof)<<std::endl; 
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
        }}
    }



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
   timerScompute = MPI_Wtime();  
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
  timerSdiagonalization = MPI_Wtime();
  if(this_mpi_process == 0)
    U = diagonalization(S,totalDimOfBasis,D);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSdiagonalization = MPI_Wtime() - timerSdiagonalization;
  pcout<<"Diagonalization of S: "<<timerSdiagonalization<<std::endl;
    MPI_Bcast(
      &(D[0]), totalDimOfBasis, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    MPI_Bcast(
      &(U[0]), totalDimOfBasis*totalDimOfBasis, MPI_DOUBLE, 0, MPI_COMM_WORLD);       

  pcout<<"Eigenvalues of S: "<<std::endl;
  for (int i = 0; i < totalDimOfBasis; i++)
    pcout<<D[i]<<" ";
  pcout<<std::endl;
  pcout<<"Eigenvectors of S: "<<std::endl;
  for(int i = 0;i < totalDimOfBasis; i++)
    {
      for(int j = 0; j < totalDimOfBasis; j++)
        pcout<<U[i*totalDimOfBasis+j]<<" ";
      pcout<<std::endl;  
    }  
  pcout<<"-------------------------"<<std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
   timerUStranspose = MPI_Wtime();
  auto Ut = TransposeMatrix(U,totalDimOfBasis);
   MPI_Barrier(MPI_COMM_WORLD);
   timerUStranspose = MPI_Wtime()-timerUStranspose;


  MPI_Barrier(MPI_COMM_WORLD);
   timerSinverse = MPI_Wtime();   
  auto invS = powerOfMatrix(-1,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSinverse = MPI_Wtime()-timerSinverse;
  pcout<<" Computing S^-1: "<<timerSinverse<<std::endl;
  pcout<<"Elements of invserse S: "<<std::endl;
  for(int i = 0;i < totalDimOfBasis; i++)
    {
      for(int j = 0; j < totalDimOfBasis; j++)
        pcout<<invS[i*totalDimOfBasis+j]<<" ";
      pcout<<std::endl;  
    }  
  pcout<<"-------------------------"<<std::endl;



  MPI_Barrier(MPI_COMM_WORLD);
  double timerSminushalf = MPI_Wtime();
  auto Sminushalf = powerOfMatrix(-0.5,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerSminushalf = MPI_Wtime() - timerSminushalf;
  

  MPI_Barrier(MPI_COMM_WORLD);
   timerShalf = MPI_Wtime();
  auto Shalf = powerOfMatrix(0.5,D,Ut,totalDimOfBasis,Ut);
  MPI_Barrier(MPI_COMM_WORLD);
  timerShalf = MPI_Wtime() - timerShalf;  
  pcout<<" Computing S^0.5: "<<timerShalf<<std::endl;
  pcout<<"Elements of  S^0.5: "<<std::endl;
  for(int i = 0;i < totalDimOfBasis; i++)
    {
      for(int j = 0; j < totalDimOfBasis; j++)
        pcout<<Shalf[i*totalDimOfBasis+j]<<" ";
      pcout<<std::endl;  
    }  
  pcout<<"-------------------------"<<std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
       timerPhiTPsi = MPI_Wtime();
      auto arrayVecOfProjserial =
        matrixTmatrixmul(scaledOrbitalValues_FEnodes,
                         n_dofs,
                         totalDimOfBasis,
                         scaledKSOrbitalValues_FEnodes,
                         n_dofs,
                         numOfKSOrbitals);
      std::vector<double> arrayVecOfProj(totalDimOfBasis * numOfKSOrbitals,
                                         0.0);
      MPI_Allreduce(&arrayVecOfProjserial[0],
                    &arrayVecOfProj[0],
                    (totalDimOfBasis * numOfKSOrbitals),
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      timerPhiTPsi = MPI_Wtime() - timerPhiTPsi;
      pcout<<"Computing PHI^TPSI: "<<timerPhiTPsi<<std::endl;
      pcout << "Matrix of projections with atomic orbitals: \n";
      MPI_Barrier(MPI_COMM_WORLD);
       timerCcompute = MPI_Wtime();
      auto coeffArrayVecOfProj = matrixmatrixmul(invS,
                                                 totalDimOfBasis,
                                                 totalDimOfBasis,
                                                 arrayVecOfProj,
                                                 totalDimOfBasis,
                                                 numOfKSOrbitals);
      MPI_Barrier(MPI_COMM_WORLD);
      timerCcompute = MPI_Wtime() - timerCcompute;  
      pcout<<"Computing C: "<<timerCcompute<<std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
       timerOcompute = MPI_Wtime();
      auto B = matrixTmatrixmul(coeffArrayVecOfProj,totalDimOfBasis,numOfKSOrbitals,S,totalDimOfBasis,totalDimOfBasis);
      auto O = matrixmatrixmul(B,numOfKSOrbitals,totalDimOfBasis,coeffArrayVecOfProj,totalDimOfBasis,numOfKSOrbitals);                                               
      MPI_Barrier(MPI_COMM_WORLD);
      timerOcompute = MPI_Wtime() -timerOcompute;
      pcout<<" Computing O: "<<timerOcompute<<std::endl;
      std::vector<double> D_O(numOfKSOrbitals,0.0);
      std::vector<double> U_O(numOfKSOrbitals*numOfKSOrbitals,0.0);
      
      MPI_Barrier(MPI_COMM_WORLD);
       timerOdiagnolaization = MPI_Wtime();
       if(this_mpi_process == 0)
	U_O =  diagonalization(O,numOfKSOrbitals,D_O);
      MPI_Barrier(MPI_COMM_WORLD);
      timerOdiagnolaization = MPI_Wtime()-timerOdiagnolaization; 
    pcout<<" Diagonalization of O: "<<timerOdiagnolaization<<std::endl;     
    MPI_Bcast(
      &(D_O[0]), numOfKSOrbitals, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    MPI_Bcast(
      &(U_O[0]), numOfKSOrbitals*numOfKSOrbitals, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
     // MPI_Barrier(MPI_COMM_WORLD);
     // timerOdiagnolaization = MPI_Wtime()-timerOdiagnolaization;

  MPI_Barrier(MPI_COMM_WORLD);
   timerUOtranspose = MPI_Wtime();
  auto U_Ot = TransposeMatrix(U_O,numOfKSOrbitals);
   MPI_Barrier(MPI_COMM_WORLD);
   timerUOtranspose = MPI_Wtime()-timerUOtranspose;  
  
  
  
      MPI_Barrier(MPI_COMM_WORLD);
       timerOminushalf = MPI_Wtime();
      auto Ominushalf = powerOfMatrix(-0.5,D_O,U_Ot,numOfKSOrbitals,U_Ot);
      MPI_Barrier(MPI_COMM_WORLD);
      timerOminushalf = MPI_Wtime() - timerOminushalf;
      pcout<<" Computing O^-0.5: "<<timerOminushalf<<std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
       timerCbarcompute = MPI_Wtime();
      std::vector<double> C_bar = matrixmatrixmul(coeffArrayVecOfProj,totalDimOfBasis,numOfKSOrbitals,
                                    Ominushalf,numOfKSOrbitals,numOfKSOrbitals);
      MPI_Barrier(MPI_COMM_WORLD);
      timerCbarcompute = MPI_Wtime() - timerCbarcompute;
      pcout<<" Computing Cbar: "<<timerCbarcompute<<std::endl;

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          writeVectorAs2DMatrix(C_bar,
                                totalDimOfBasis,
                                numOfKSOrbitals,
                                "FePOP_v1.txt");
          // printVector(CoeffofOrthonormalisedKSonAO);
        }
      MPI_Barrier(MPI_COMM_WORLD);
       timerChatcompute = MPI_Wtime();
      std::vector<double> C_hat = matrixmatrixmul(Shalf,totalDimOfBasis,totalDimOfBasis,C_bar,totalDimOfBasis,numOfKSOrbitals);
      MPI_Barrier(MPI_COMM_WORLD);
      timerChatcompute = MPI_Wtime() - timerChatcompute; 
      pcout<<" Computing Chat: "<<timerChatcompute<<std::endl;
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          writeVectorAs2DMatrix(C_hat,
                                totalDimOfBasis,
                                numOfKSOrbitals,
                                "FePHP_v1.txt");
        }

        MPI_Barrier(MPI_COMM_WORLD);
         timerHprojOrbital = MPI_Wtime();
        auto Hproj_orbital = computeHprojOrbital(C_hat,C_hat,
                                               totalDimOfBasis,
                                               numOfKSOrbitals,
                                               eigenValues[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        timerHprojOrbital = MPI_Wtime()-timerHprojOrbital;                                       
        pcout<<" Computing Projected Hamiltonian: "<<timerHprojOrbital<<std::endl;
      
      
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          writeVectorAs2DMatrix(Hproj_orbital,
                                totalDimOfBasis,
                                totalDimOfBasis,
                                "Hproj_orbital.txt");
        }
      // Compute projected Hamiltonian of FE discretized Hamiltonian into
      
        Hproj_orbital.clear();
	C_hat.clear();
	C_bar.clear();	
	O.clear();
	U_O.clear();
	B.clear();
	
	std::vector<double>().swap(Hproj_orbital);
	std::vector<double>().swap(C_hat);
	std::vector<double>().swap(C_bar);
	std::vector<double>().swap(O);
	std::vector<double>().swap(U_O);
	std::vector<double>().swap(B);

	pcout
        << "--------------------------COHP Data Saved------------------------------"
        << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      if (this_mpi_process == 0)
        {
          // writing the energy levels and the occupation numbers
          unsigned int  kPointDummy = 0;
          std::ofstream energyLevelsOccNumsFile("energyLevelsOccNums.txt");

          if (energyLevelsOccNumsFile.is_open())
            {
              for (unsigned int i = 0; i < eigenValues[0].size(); ++i)
                {
                  const double partialOccupancy =
                    dftUtils::getPartialOccupancy(eigenValues[kPointDummy][i],
                                                  fermiEnergy,
                                                  C_kb,
                                                  d_dftParamsPtr->TVal);

                  energyLevelsOccNumsFile << eigenValues[kPointDummy][i] << " "
                                          << partialOccupancy << '\n';
                }

              energyLevelsOccNumsFile.close();
            }

          else
            pcout << "couldn't open energyLevelsOccNums.txt file!\n";
        }
      if (this_mpi_process == 0)
        {
          pcout
            << "\n-------------------------------------------------------\n";
          pcout << "Projected SpillFactors are:" << std::endl;
          spillFactorsofProjectionwithCS(coeffArrayVecOfProj,
                                         upperTriaOfS,
                                         occupationNum,
                                         totalDimOfBasis,
                                         numOfKSOrbitals,
                                         totalDimOfBasis,
                                         totalDimOfBasis);
          pcout
            << "\n-------------------------------------------------------\n";
        }
#endif 
  pcout<<"----------------------------------------------------------"<<std::endl;
  pcout<<"------------------- OLD METHOD ----------------------------"<<std::endl;
  pcout<<" Creating PHI and PSI matrices: "<<timerCreatingMatrices<<std::endl;
  pcout<<" Computing S matrix: "<<timerScompute<<std::endl;
  pcout<<"Computing PHI^TPSI: "<<timerPhiTPsi<<std::endl;
  pcout<<"Diagonalization of S: "<<timerSdiagonalization<<std::endl;
  pcout<<" Computing S^-1: "<<timerSinverse<<std::endl;
  pcout<<"Computing C: "<<timerCcompute<<std::endl;
  pcout<<" Computing O: "<<timerOcompute<<std::endl;
  pcout<<" Diagonalization of O: "<<timerOdiagnolaization<<std::endl;
  pcout<<" Computing O^-0.5: "<<timerOminushalf<<std::endl;
  pcout<<" Computing Cbar: "<<timerCbarcompute<<std::endl;
  pcout<<" Computing S^0.5: "<<timerShalf<<std::endl;
  pcout<<" Computing Chat: "<<timerChatcompute<<std::endl;
  pcout<<" Computing Projected Hamiltonian: "<<timerHprojOrbital<<std::endl;
  pcout<<" Compute U transpose: "<<timerUStranspose<<std::endl ;
  pcout<<" Compute U_O transpose: "<<timerUOtranspose<<std::endl ;
  pcout<<" TOTAL TIME in sec: "<<timerScompute+timerPhiTPsi+timerSdiagonalization+timerSinverse+
  timerCcompute+timerOcompute+timerOdiagnolaization+timerOminushalf+timerCbarcompute+timerShalf+timerChatcompute+
  timerHprojOrbital<<std::endl;
  pcout<<"----------------------------------------------------------"<<std::endl;
  pcout<<std::endl;
   pcout<<"----------------------------------------------------------"<<std::endl;


    
}
