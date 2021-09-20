// using this class we create an Array of Objects
// // each element corresponding to an atom type
// // corresponding atomPositions should be as an external array or suitable in a datastructure

void constructQuantumNumbersHierarchy
		(unsigned int nstart, unsigned int nend, 
		 std::vector<OrbitalQuantumNumbers>& quantumNumHierarchy)
{
     // assume the vector of size 0 has already been reserved with space for N shells 
     // which is N(N+1)(2N+1)/6 orbitals for N shells 
     // N is maximum of the principal quantum number over each atomType
     // this function is called just once for the whole program 
  
      	OrbitalQuantumNumbers orbitalTraverse;

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
}



template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::orbitalOverlapPopulationCompute()
{
	std::cout << std::fixed;
	std::cout << std::setprecision(8);

	std::cout << "Started post-processing DFT results to obtain Bonding information..\n";

	// would it be good to replace (unsigned) int with (unsigned) short int?
	// would it be better to replace unsigned short int with uint16_t?

	//**************** Forming data structures of atom info ****************// 

	std::cout << "reading input files..\n";

	// for Hydrogen molecule: 
	// const unsigned int numOfAtoms = 2;
	// const unsigned int numOfAtomTypes = 1; 

	// for CO molecule: 
	const unsigned int numOfAtoms = 2;
	const unsigned int numOfAtomTypes = 2; // is not more than 120

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

	std::string coordinatesFile = "coordinatesNew.inp"; // In DFT-FE format

	std::ifstream atomCoordinatesFile (coordinatesFile);
	if (atomCoordinatesFile.is_open()){
		while(atomCoordinatesFile >> a >> b >> x >> y >> z){

			atomicNumVec.push_back(a);
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

	std::cout << "reading " << coordinatesFile << " complete!\n";

	assert(count == numOfAtoms); // we can add an assert message later 
	assert(numOfAtomTypes == atomTypesSet.size());

	std::vector<unsigned int> atomTypesVec {atomTypesSet.begin(), atomTypesSet.end()};

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

	unsigned int maxBasisShell = 0; // some lower unreachable value 
	// unsigned int minBasisShell = 10; // some unreachable value

	unsigned int tmpBasisDim, b1, b2;

	// b1 and b2 are the range of principal quantum numbers, ends inclusive
	// where the slater type basis are constructed based on n, l, m values or each orbital 

	// at present zeta is fixed for the atom but soon we ll have 
	// to construct it on own using the slater rules function 

	std::string basisInfoFile = "STOBasisInfoNew.inp";

	std::ifstream basisinfofile (basisInfoFile);
	if (basisinfofile.is_open()){
		while(basisinfofile >> a >> b1 >> b2 >> zeta){

			tmpBasisDim = numOfOrbitalsForShellCount(b1, b2);

			atomTypeswithBasisInfo.insert(a);
			atomTypetoBasisDim.insert({a, tmpBasisDim});
			atomTypetoZeta.insert({a, zeta});

			// observe that all the above are from atomType and not atomTypeID or atomID
			// principal quantum number range 

			atomTypetoNstart.insert({a, b1});
			atomTypetoNend.insert({a, b2});

			// to construct quantum number hierarchy 

			maxBasisShell = std::max(maxBasisShell, b2);

			++count;
		}
	}

	else {

		std::cerr << "Couldn't open " << basisInfoFile << " file!!" << std::endl;
        exit(0);
	}

	basisinfofile.close();
	std::cout << "reading " << basisInfoFile << " complete!\n";

	assert(count == numOfAtomTypes);

	std::cout << "reading and storing inputs done!\n";

	std::set_difference(atomTypesSet.begin(), atomTypesSet.end(), 
		atomTypeswithBasisInfo.begin(), atomTypeswithBasisInfo.end(), 
		std::inserter(diff, diff.begin()));
	assert(diff.empty()); 

	atomTypesSet.clear();
	atomTypeswithBasisInfo.clear();

	// based on the elements in diff set we can ask the user 
	// to update the info for those specific atoms  

	// we need a function now to convert between global basis numbering
	// and atom ID (as per the coordinates, not atom type ID, and atomic number) 
	// and its local basis numbering 

	// by the way can we automatically have an inverse function from a defined function
	// if it is given that that the function is one-one and onto  -- seems not easy! 

	count = 0; 
	std::vector<unsigned int> atomwiseGlobalbasisNum; // cumulative vector  
	atomwiseGlobalbasisNum.reserve(numOfAtoms + 1);

	atomwiseGlobalbasisNum.push_back(count); // first entry is zero 
	
	for (auto& i : atomicNumVec) // atomicNumVec has entry for each atom, with atomType repetitions 
	{
		count += atomTypetoBasisDim[i];	
		atomwiseGlobalbasisNum.push_back(count);
	}

	unsigned int totalDimOfBasis = count; 

	std::cout << "total basis dimension: " << totalDimOfBasis << '\n'
			  << "total number of atoms: " << numOfAtoms << '\n'
			  << "number of atoms types: " << numOfAtomTypes << '\n'
			  << "max basis shell: " << maxBasisShell << '\n';

	std::vector<LocalAtomicBasisInfo> globalBasisInfo;
	globalBasisInfo.reserve(totalDimOfBasis);
	unsigned int tmp1, tmp2, tmp3, nstart, basisNstart;

	for(unsigned int i = 0; i < numOfAtoms; ++i) {

		tmp1 = atomTypetoAtomTypeID[ atomicNumVec[i] ];
		tmp2 =  atomwiseGlobalbasisNum[i];
		tmp3 = atomwiseGlobalbasisNum[i+1];
		nstart = atomTypetoNstart[ atomicNumVec[i] ];
		basisNstart = numOfOrbitalsForShellCount(1, nstart - 1);

		for(unsigned int j = tmp2; j < tmp3; ++j)
		{
			globalBasisInfo.push_back({i, tmp1, basisNstart + j - tmp2}); 
			// i required to get atom position coordinates 
			// atomTypeID is required to construct the basis 
			// basisNum in the quantumNumHierarchy
		}
	}

	std::cout << "global basis info constructed!\n";

	std::vector<OrbitalQuantumNumbers> quantumNumHierarchy;
	quantumNumHierarchy.reserve(numOfOrbitalsForShellCount(1, maxBasisShell));
	
	constructQuantumNumbersHierarchy(1, maxBasisShell, quantumNumHierarchy);

	std::cout << "quantum hierarchy constructed!\n";

	// vector of AtomicOrbitalBasisManager objects 

	std::vector<AtomicOrbitalBasisManager> atomTypewiseSTOvector;
	atomTypewiseSTOvector.reserve(numOfAtomTypes);
	int atomType;

	for(unsigned int i = 0; i < numOfAtomTypes; ++i)
	{
		atomType = atomTypesVec[i];
		atomTypewiseSTOvector.push_back(AtomicOrbitalBasisManager
			(atomType, 1, true, atomTypetoBasisDim[atomType], atomTypetoZeta[atomType]));
	}

	std::cout << "vector of objects constructed!\n";

	//***************** Atom info datastructures constructed ****************// 

	//************ Constucting FE mass matrix for GL Quadrature *************// 

	// Initial setup done!

	// Generating a fully functioning FE dealii framework 
	// constructing Mass matrix - Only the diagonal elements would be present
	// cross terms overlap vanishes as the FE nodes are present GL quadrature points  

	//**************************** FE Framework *****************************//

	/*std::cout << "constructing FE framework.. " << '\n';

	const unsigned int dim = 3; // physical dimension of the problem 

	// const std::vector<unsigned int> uniform_grid {20, 20, 20}; // coarse mesh 
	const std::vector<unsigned int> uniform_grid {40, 40, 40}; // testing purpose - usual choice
	// const std::vector<unsigned int> uniform_grid {60, 60, 60}; // finer mesh 

	const Point<dim> p1 (-7.0, -7.0, -7.0);
	const Point<dim> p2 (+7.0, +7.0, +7.0); 

	Triangulation<dim> triangulation;

	GridGenerator::subdivided_hyper_rectangle(triangulation, uniform_grid, p1, p2);

	unsigned int feOrder = 3; //  = 3, 4, 5; or else reduce mess size by half 

	DoFHandler<3> dof_handler;	// dof_handler for connectivity

	dealii::types::global_dof_index  n_dofs, n_q_points, dofs_per_cell;

	FE_Q<dim> FE(feOrder); // By default it creates finite-element object with nodes located at Gauss Lobatto  points	

	// Initialize dof_handler

	dof_handler.initialize(triangulation, FE);
	dof_handler.distribute_dofs(FE);

	// to access the indices of locally owned degrees of freedom
	// locallly owned in the sense of distributed memory parallelism (MPI)

	const IndexSet &locallyOwnedSet = dof_handler.locally_owned_dofs();
	std::vector<IndexSet::size_type> locallyOwnedDOFs;
	locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
	n_dofs = locallyOwnedDOFs.size(); // Number of finite-element nodes in the mesh

	// Constructing dealii vector corresponding to diagonal of a mass matrix //

	// create a dealii vector

	Vector<double> sqrtMassVector(n_dofs);

	// create a constraintsMatrix (Here it is just dummy for now)

	AffineConstraints<double> constraintsDummy;
	DoFTools::make_hanging_node_constraints(dof_handler, constraintsDummy);

	// declare Gauss-Lobatto quadrature rule
	QGaussLobatto<3> quadrature(feOrder + 1);
	FEValues<3> fe_framework(dof_handler.get_fe(),
                      		 quadrature, update_values | update_JxW_values);

	// need to switch to Gauss Quadrature or may be have a different FE values setup 
	// if required 

	std::cout << "constructing FE framwork done!\n";

	//***************** Constucting FE triangulation completed *****************// 

	  /*std::cout << "constructing Mass matrix using Gauss-Lobatto quadrature.. \n";
					  
	dofs_per_cell   = (dof_handler.get_fe()).dofs_per_cell;
	n_q_points = quadrature.size(); // redefininition of n_q_points 

	// Create dealii vector of size dofs_per_cell
	Vector<double> massVectorLocal(dofs_per_cell);
	std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);	

	typename DoFHandler<3>::active_cell_iterator cell = dof_handler.begin_active(),
                                                    endc = dof_handler.end();

	for (; cell != endc; ++cell)
      	if (cell->is_locally_owned()) // declare cell type 
        {
          	// compute values for the current element
          	fe_framework.reinit(cell);
          	massVectorLocal = 0.0;
          	for (unsigned int i = 0; i < dofs_per_cell; ++i)
            	for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              		massVectorLocal(i) += fe_framework.shape_value(i, q_point) *
                                    	  fe_framework.shape_value(i, q_point) *
                                    	  fe_framework.JxW(q_point);

          	cell->get_dof_indices(local_dof_indices);
          	constraintsDummy.distribute_local_to_global(massVectorLocal,local_dof_indices, sqrtMassVector);
        }
		
	sqrtMassVector.compress(VectorOperation::add);

	for(dealii::types::global_dof_index i = 0; i < sqrtMassVector.size(); ++i)
	{
   		if (sqrtMassVector.in_local_range(i) &&
          	!constraintsDummy.is_constrained(i))
   		sqrtMassVector(i) = std::sqrt(sqrtMassVector(i));
	}
	sqrtMassVector.compress(VectorOperation::insert); // this is now the sqrt of mass vector

	std::cout << "Mass matrix (diagonal vector) constructed! \n";*/
	// Mass vector constructed! 

	//******************* Mass diagonal matrix constructed ******************// 

	// evaluating projections and coefficients of projections onto Atomic orbitals 

	// unsigned int numOfKSOrbitals = 1; // For Hydrogen molecule case  
	unsigned int numOfKSOrbitals = 8; // For CO molecule case  

	std::cout << "Number of Kohn-Sham orbitals: " << numOfKSOrbitals << '\n';


	std::vector<std::function<double(const dealii::Point<3>)>> MOsOfCO; 
	std::vector<double> energyLevelsKS;
	std::vector<int> occupationNum;

	// a test value for Hydrogen Bonding orbital
	// energyLevelsKS.push_back(0.0);  
	// occupationNum.push_back(2);

	assembleCO_LCAO_MOorbitals(energyLevelsKS, MOsOfCO, occupationNum); // for CO molecule

	// access nodal point data to evaluate the functions 

	//std::map<dealii::types::global_dof_index, Point<3>> d_supportPoints;
	//DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(), dof_handler, d_supportPoints);

	// Now create a flattened STL vector of size (number of dofs) x (number of orbitals per atom) let’s say this STL vector is called “atomicOrbital” and compute finite-element nodal values and fill in this STL vector

	// Loop over atomic orbitals to evaluate at all nodal points
	
	const IndexSet &locallyOwnedSet = dofHandlerEigen.locally_owned_dofs();
	std::vector<IndexSet::size_type> locallyOwnedDOFs;
	locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
	unsigned int n_dofs = locallyOwnedDOFs.size();
	std::vector<double> scaledOrbitalValues_FEnodes(n_dofs * totalDimOfBasis, 0.0);
	std::vector<double> scaledKSOrbitalValues_FEnodes(n_dofs * numOfKSOrbitals, 0.0);
 
	for (unsigned int dof = 0; dof < n_dofs; ++dof)
	{
		// get nodeID 
    	const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
    
    	// get coordinates of the finite-element node
    	Point<3> node  = d_supportPointsEigen[dofID];

    	auto count1 = totalDimOfBasis*dof;

		for (unsigned int i = 0; i < totalDimOfBasis; ++i)
		{
			auto atomPos = atomCoordinates[ globalBasisInfo[i].atomID ];
  			auto atomTypeID = globalBasisInfo[i].atomTypeID;
  			auto orbital = quantumNumHierarchy[ globalBasisInfo[i].localBasisNum ];

			scaledOrbitalValues_FEnodes[count1 + i] = d_kohnShamDFTOperatorPtr->sqrtMassVector[dof] *
								atomTypewiseSTOvector[atomTypeID].hydrogenicOrbital
  									(orbital, node, atomPos);
		}

		auto count2 = numOfKSOrbitals*dof;

		for (unsigned int j = 0; j < numOfKSOrbitals; ++j)
		{
			scaledKSOrbitalValues_FEnodes[count2 + j] =  d_kohnShamDFTOperatorPtr->sqrtMassVector[dof] *  MOsOfCO[j](node);
														 // hydrogenMoleculeBondingOrbital(node);  MOsOfCO[j](node);
		}

	}

	// matrix of orbital values at FE nodes constructed!
	//std::cout << "matrices of orbital values at the nodes constructed!\n";

	// direct assembly of Overlap matrix S using Mass diagonal matrix from Gauss Lobatto

	// actually we must write such that only the symmetric upper triangular part is calculated
	// observe in the above we used same variable at two arguments in the function
	// and this function have those arguments in const referenced way! Let's see if it works.. 

	auto upperTriaOfS = selfMatrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis);
	std::cout << "Upper triangular part of Overlap matrix (S) vector in the direct way: \n";
	printVector(upperTriaOfS);

	auto invS = inverseOfOverlapMatrix(upperTriaOfS, totalDimOfBasis);
	std::cout << "Full S inverse matrix: \n";
	printVector(invS);

	auto arrayVecOfProj 
					= matrixTmatrixmul(scaledOrbitalValues_FEnodes, n_dofs, totalDimOfBasis, 
									   scaledKSOrbitalValues_FEnodes, n_dofs, numOfKSOrbitals);

	std::cout << "Matrix of projections with atomic orbitals: \n";
	printVector(arrayVecOfProj);

	auto coeffArrayVecOfProj
					= matrixmatrixmul(invS, 		  totalDimOfBasis, totalDimOfBasis, 
									  arrayVecOfProj, totalDimOfBasis, numOfKSOrbitals);

	std::cout << "Matrix of coefficients of projections: \n";
	printVector(coeffArrayVecOfProj);

	auto spilling = spillFactorsOfProjection(coeffArrayVecOfProj, arrayVecOfProj, occupationNum);

	std::cout << "Total spilling = " << spilling.totalSpilling << '\n';
	std::cout << "Absolute total spilling = " << spilling.absTotalSpilling << '\n';
	std::cout << "Charge Spilling = " << spilling.chargeSpilling << '\n';
	std::cout << "Absolute charge spilling = " << spilling.absChargeSpilling << '\n';
}
