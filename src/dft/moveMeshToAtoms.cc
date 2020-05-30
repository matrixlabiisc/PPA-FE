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

	template<unsigned int FEOrder>
void dftClass<FEOrder>::calculateNearestAtomDistances()
{
	const unsigned int numberGlobalAtoms = atomLocations.size();
	const unsigned int numberImageAtoms = d_imageIdsTrunc.size();  
  d_nearestAtomDistances.clear();
  d_nearestAtomIds.clear();
  d_nearestAtomDistances.resize(numberGlobalAtoms,1e+6);
  d_nearestAtomIds.resize(numberGlobalAtoms);
	for (unsigned int i=0;i <numberGlobalAtoms; i++)
  {
    Point<3> atomCoori;
    Point<3> atomCoorj;
    atomCoori[0] = atomLocations[i][2];
    atomCoori[1] = atomLocations[i][3];
    atomCoori[2] = atomLocations[i][4];
		for (unsigned int j=0;j <(numberGlobalAtoms+numberImageAtoms); j++)
		{
      if(j < numberGlobalAtoms)
      {
        atomCoorj[0] = atomLocations[j][2];
        atomCoorj[1] = atomLocations[j][3];
        atomCoorj[2] = atomLocations[j][4];
      }
      else
      {
        atomCoorj[0] = d_imagePositionsTrunc[j-numberGlobalAtoms][0];
        atomCoorj[1] = d_imagePositionsTrunc[j-numberGlobalAtoms][1];
        atomCoorj[2] = d_imagePositionsTrunc[j-numberGlobalAtoms][2];
      }     

			const double dist=atomCoori.distance(atomCoorj);
			if (dist<d_nearestAtomDistances[i] && j!=i)
      {
				d_nearestAtomDistances[i] =dist;
        d_nearestAtomIds[i]=j;
      }
		}
  }
  d_minDist=*std::min_element(d_nearestAtomDistances.begin(),d_nearestAtomDistances.end());
	if (dftParameters::verbosity>=2)
		pcout<<"Minimum distance between atoms: "<<d_minDist<<std::endl;

	AssertThrow(d_minDist>=0.1,ExcMessage("DFT-FE Error: Minimum distance between atoms is less than 0.1 Bohr."));   
}


	template<unsigned int FEOrder>
void dftClass<FEOrder>::moveMeshToAtoms(Triangulation<3,3> & triangulationMove,
		Triangulation<3,3> & triangulationSerial,
		bool reuseClosestTriaVertices,
		bool moveSubdivided)
{
	dealii::ConditionalOStream pcout_movemesh (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
	dealii::TimerOutput timer_movemesh(mpi_communicator,
			pcout_movemesh,
			dftParameters::reproducible_output ||
			dftParameters::verbosity<4 ? dealii::TimerOutput::never:
			dealii::TimerOutput::summary,dealii::TimerOutput::wall_times);

	meshMovementGaussianClass gaussianMove(mpi_communicator);
	gaussianMove.init(triangulationMove,
			triangulationSerial,
			d_domainBoundingVectors);

	const unsigned int numberGlobalAtoms = atomLocations.size();
	const unsigned int numberImageAtoms = d_imageIdsTrunc.size();

	std::vector<Point<3>> atomPoints;
	d_atomLocationsAutoMesh.resize(numberGlobalAtoms,std::vector<double>(3,0.0));
	for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
	{
		Point<3> atomCoor;
		atomCoor[0] = atomLocations[iAtom][2];
		atomCoor[1] = atomLocations[iAtom][3];
		atomCoor[2] = atomLocations[iAtom][4];
		atomPoints.push_back(atomCoor);
		for (unsigned int j=0;j <3; j++)
			d_atomLocationsAutoMesh[iAtom][j]=atomCoor[j];
	}

	std::vector<Point<3>> closestTriaVertexToAtomsLocation;
	std::vector<Tensor<1,3,double> > dispClosestTriaVerticesToAtoms;

	timer_movemesh.enter_section("move mesh to atoms: find closest vertices");
	if(reuseClosestTriaVertices)
	{
		closestTriaVertexToAtomsLocation = d_closestTriaVertexToAtomsLocation;
		dispClosestTriaVerticesToAtoms = d_dispClosestTriaVerticesToAtoms;
	}
	else
	{
		gaussianMove.findClosestVerticesToDestinationPoints(atomPoints,
				closestTriaVertexToAtomsLocation,
				dispClosestTriaVerticesToAtoms);
	}
	timer_movemesh.exit_section("move mesh to atoms: find closest vertices");


	timer_movemesh.enter_section("move mesh to atoms: move mesh");
	//add control point locations and displacements corresponding to images
	if(!reuseClosestTriaVertices)
		for(unsigned int iImage=0;iImage <numberImageAtoms; iImage++)
		{
			Point<3> imageCoor;
			Point<3> correspondingAtomCoor;

			imageCoor[0] = d_imagePositionsTrunc[iImage][0];
			imageCoor[1] = d_imagePositionsTrunc[iImage][1];
			imageCoor[2] = d_imagePositionsTrunc[iImage][2];
			const int atomId=d_imageIdsTrunc[iImage];
			correspondingAtomCoor[0] = atomLocations[atomId][2];
			correspondingAtomCoor[1] = atomLocations[atomId][3];
			correspondingAtomCoor[2] = atomLocations[atomId][4];


			const dealii::Point<3> temp=closestTriaVertexToAtomsLocation[atomId]+(imageCoor-correspondingAtomCoor);
			closestTriaVertexToAtomsLocation.push_back(temp);
			dispClosestTriaVerticesToAtoms.push_back(dispClosestTriaVerticesToAtoms[atomId]);
		}

	d_closestTriaVertexToAtomsLocation = closestTriaVertexToAtomsLocation;
	d_dispClosestTriaVerticesToAtoms = dispClosestTriaVerticesToAtoms;
	d_gaussianMovementAtomsNetDisplacements.resize(numberGlobalAtoms);
	for(unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
		d_gaussianMovementAtomsNetDisplacements[iAtom]=0.0;

	d_controlPointLocationsCurrentMove.clear();
  d_gaussianConstantsAutoMesh.clear();
  d_flatTopWidthsAutoMeshMove.clear();
  d_gaussianConstantsAutoMesh.resize(numberGlobalAtoms);
  d_flatTopWidthsAutoMeshMove.resize(numberGlobalAtoms);
	for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
	{
      d_flatTopWidthsAutoMeshMove[iAtom]=dftParameters::useFlatTopGenerator?std::min(0.5,0.9*d_nearestAtomDistances[iAtom]/2.0-0.4):0.0;
      d_gaussianConstantsAutoMesh[iAtom]=dftParameters::reproducible_output?1/std::sqrt(0.5):(std::min(0.9* d_nearestAtomDistances[iAtom]/2.0, 2.0)-d_flatTopWidthsAutoMeshMove[iAtom]);
  }

  std::vector<double> gaussianConstantsAutoMesh;
  std::vector<double> flatTopWidths;
	for (unsigned int iAtom=0;iAtom <numberGlobalAtoms+numberImageAtoms; iAtom++)
	{
		Point<3> atomCoor;
		if(iAtom < numberGlobalAtoms)
		{
			atomCoor[0] = atomLocations[iAtom][2];
			atomCoor[1] = atomLocations[iAtom][3];
			atomCoor[2] = atomLocations[iAtom][4];
      d_gaussianConstantsAutoMesh.push_back(dftParameters::reproducible_output?1/std::sqrt(0.5):std::min(0.9* d_nearestAtomDistances[iAtom]/2.0, 2.0));
		}
		else
		{
			atomCoor[0] = d_imagePositionsTrunc[iAtom-numberGlobalAtoms][0];
			atomCoor[1] = d_imagePositionsTrunc[iAtom-numberGlobalAtoms][1];
			atomCoor[2] = d_imagePositionsTrunc[iAtom-numberGlobalAtoms][2];
		}
		d_controlPointLocationsCurrentMove.push_back(atomCoor);
	}

	for (unsigned int iAtom=0;iAtom <numberGlobalAtoms+numberImageAtoms; iAtom++)
	{
    int atomId;
		if(iAtom < numberGlobalAtoms)
      atomId=iAtom;
		else
      atomId=d_imageIdsTrunc[iAtom-numberGlobalAtoms];
		
    gaussianConstantsAutoMesh.push_back(d_gaussianConstantsAutoMesh[atomId]);
    flatTopWidths.push_back(d_flatTopWidthsAutoMeshMove[atomId]);
	}

	const std::pair<bool,double> meshQualityMetrics=gaussianMove.moveMesh(closestTriaVertexToAtomsLocation,
			dispClosestTriaVerticesToAtoms,
			gaussianConstantsAutoMesh,
			flatTopWidths,
			moveSubdivided);

	timer_movemesh.exit_section("move mesh to atoms: move mesh");

	AssertThrow(!meshQualityMetrics.first,ExcMessage("Negative jacobian created after moving closest nodes to atoms. Suggestion: increase refinement near atoms"));

	if(!reuseClosestTriaVertices)
		d_autoMeshMaxJacobianRatio = meshQualityMetrics.second;

	if (dftParameters::verbosity>=1 && !moveSubdivided)
		pcout<< "Mesh quality check for Auto mesh after mesh movement, maximum jacobian ratio: "<< meshQualityMetrics.second<<std::endl;

  calculateAdaptiveForceGeneratorsSmearedChargeWidths();  
}

	template<unsigned int FEOrder>
void dftClass<FEOrder>::calculateAdaptiveForceGeneratorsSmearedChargeWidths()
{
  d_gaussianConstantsForce.clear();
  d_generatorFlatTopWidths.clear();
  d_smearedChargeWidths.clear();
  const unsigned int vertices_per_cell=dealii::GeometryInfo<3>::vertices_per_cell;

	const unsigned int numberGlobalAtoms = atomLocations.size();
	const unsigned int numberImageAtoms = d_imageIdsTrunc.size();

  d_gaussianConstantsForce.resize(numberGlobalAtoms);
  d_generatorFlatTopWidths=d_flatTopWidthsAutoMeshMove;
	for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
      d_gaussianConstantsForce[iAtom]=dftParameters::reproducible_output?1/std::sqrt(5.0):(dftParameters::useFlatTopGenerator?d_generatorFlatTopWidths[iAtom]+0.4:(std::min(0.9*d_nearestAtomDistances[iAtom]/2.0-0.3,dftParameters::gaussianConstantForce)));  
  d_smearedChargeWidths.resize(numberGlobalAtoms,0.0);

  if (dftParameters::smearedNuclearCharges)
  {
    const double gaussianWidth=0.3;
    const double maxFlatTopWidth=1.5;
    std::vector<double> generatorFlatTopWidthsAllAtomsImages(numberGlobalAtoms+numberImageAtoms,0.0);
    std::vector<double> smearedChargeWidthsAllAtomsImages(numberGlobalAtoms+numberImageAtoms,0.0);

    dealii::BoundingBox<3> boundingBoxTria(vectorTools::createBoundingBoxTriaLocallyOwned(dofHandler));
    dealii::Tensor<1,3,double> tempDisp;
    tempDisp[0]=2.0;
    tempDisp[1]=2.0;
    tempDisp[2]=2.0;

    for (unsigned int iAtom=0;iAtom <numberGlobalAtoms+numberImageAtoms; iAtom++)
    {
      Point<3> atomCoord;
      unsigned int atomId;
      if(iAtom < numberGlobalAtoms)
      {
        atomId=iAtom;
        atomCoord[0] = atomLocations[iAtom][2];
        atomCoord[1] = atomLocations[iAtom][3];
        atomCoord[2] = atomLocations[iAtom][4];
      }
      else
      {
        const int atomId=d_imageIdsTrunc[iAtom-numberGlobalAtoms];
        atomCoord[0] = d_imagePositionsTrunc[iAtom-numberGlobalAtoms][0];
        atomCoord[1] = d_imagePositionsTrunc[iAtom-numberGlobalAtoms][1];
        atomCoord[2] = d_imagePositionsTrunc[iAtom-numberGlobalAtoms][2];
      }


      std::pair< dealii::Point<3,double >,dealii::Point<3, double>> boundaryPoints;
      boundaryPoints.first=atomCoord-tempDisp;
      boundaryPoints.second=atomCoord+tempDisp;
      dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

      if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom)==NeighborType::not_neighbors)
        continue;

      smearedChargeWidthsAllAtomsImages[iAtom]=0.4;
    
      int count=0;
      double boundingSphereRadius=generatorFlatTopWidthsAllAtomsImages[iAtom];
      while ((boundingSphereRadius*2.0+gaussianWidth*2.0)>d_nearestAtomDistances[atomId] || (boundingSphereRadius>maxFlatTopWidth) || count==0)
      {
        boundingSphereRadius=0.0;
        DoFHandler<3>::active_cell_iterator
          cell = dofHandler.begin_active(),
               endc = dofHandler.end();
        for(; cell!=endc; ++cell) 
          if(cell->is_locally_owned())
          {
              double maximumDistanceFromVertex=0.0;
              double minimumDistanceFromVertex=1e+6;            
              for (unsigned int i=0; i<vertices_per_cell; ++i)
              {
                dealii::Point<3> nodalCoor = cell->vertex(i);
                const double dist=atomCoord.distance(nodalCoor);
                if (dist>maximumDistanceFromVertex)
                  maximumDistanceFromVertex=dist;
                                
                if (dist<minimumDistanceFromVertex)
                  minimumDistanceFromVertex=dist;
              } 

              if (minimumDistanceFromVertex<smearedChargeWidthsAllAtomsImages[iAtom] && boundingSphereRadius<maximumDistanceFromVertex)
                boundingSphereRadius=maximumDistanceFromVertex;
          }

        boundingSphereRadius+=1e-3;
        generatorFlatTopWidthsAllAtomsImages[iAtom]=boundingSphereRadius;

        if ((boundingSphereRadius*2.0+gaussianWidth*2)>d_nearestAtomDistances[atomId])
          smearedChargeWidthsAllAtomsImages[iAtom]-=0.05;

	      AssertThrow(smearedChargeWidthsAllAtomsImages[iAtom]>=0.05,ExcMessage("DFT-FE Error: Smeared charge width calculated adaptively is less than 0.05 Bohr. Hint: Refine the mesh further for regions where atoms are close (<2 Bohr).")); 
        count++;
      }
    }
   
		MPI_Allreduce(MPI_IN_PLACE,
					&smearedChargeWidthsAllAtomsImages[0],
					numberGlobalAtoms+numberImageAtoms,
					MPI_DOUBLE,
					MPI_MAX,
					mpi_communicator);

		MPI_Allreduce(MPI_IN_PLACE,
					&generatorFlatTopWidthsAllAtomsImages[0],
					numberGlobalAtoms+numberImageAtoms,
					MPI_DOUBLE,
					MPI_MAX,
					mpi_communicator);    

    MPI_Bcast(&smearedChargeWidthsAllAtomsImages[0],
        numberGlobalAtoms+numberImageAtoms,
        MPI_DOUBLE,
        0,
        interpoolcomm);

    MPI_Bcast(&smearedChargeWidthsAllAtomsImages[0],
        numberGlobalAtoms+numberImageAtoms,
        MPI_DOUBLE,
        0,
        interBandGroupComm); 

    MPI_Bcast(&generatorFlatTopWidthsAllAtomsImages[0],
        numberGlobalAtoms+numberImageAtoms,
        MPI_DOUBLE,
        0,
        interpoolcomm);

    MPI_Bcast(&generatorFlatTopWidthsAllAtomsImages[0],
        numberGlobalAtoms+numberImageAtoms,
        MPI_DOUBLE,
        0,
        interBandGroupComm);     

    for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
    {
      d_generatorFlatTopWidths[iAtom]=generatorFlatTopWidthsAllAtomsImages[iAtom];
      d_smearedChargeWidths[iAtom]=smearedChargeWidthsAllAtomsImages[iAtom];      
    }

    for (unsigned int iAtom=0;iAtom <numberImageAtoms; iAtom++)
    {
      const unsigned atomId=d_imageIdsTrunc[iAtom];
        
      if (generatorFlatTopWidthsAllAtomsImages[iAtom+numberGlobalAtoms]>d_generatorFlatTopWidths[atomId])
        d_generatorFlatTopWidths[atomId]=generatorFlatTopWidthsAllAtomsImages[iAtom+numberGlobalAtoms];

      if (smearedChargeWidthsAllAtomsImages[iAtom+numberGlobalAtoms]<d_smearedChargeWidths[atomId] && smearedChargeWidthsAllAtomsImages[iAtom+numberGlobalAtoms]>1e-3)
        d_smearedChargeWidths[atomId]=smearedChargeWidthsAllAtomsImages[iAtom+numberGlobalAtoms];      
    }

    for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
        d_gaussianConstantsForce[iAtom]=std::min(gaussianWidth,maxFlatTopWidth+gaussianWidth-d_generatorFlatTopWidths[iAtom]);

    if (dftParameters::verbosity>=4)
      for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
          pcout<< "iAtom: "<< iAtom<<", Smeared charge width: "<<d_smearedChargeWidths[iAtom]<< ",  flat top width: "<< d_generatorFlatTopWidths[iAtom] <<std::endl;        
  }
}
