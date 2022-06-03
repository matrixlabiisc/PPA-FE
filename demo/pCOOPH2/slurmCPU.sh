#!/bin/bash
#SBATCH --job-name Test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=180g
#SBATCH --time=10:00:00
#SBATCH --account=vikramg1
#SBATCH --mail-user=phanim@umich.edu
#SBATCH --mail-type=BEGIN,END

#srun /scratch/vikramg_root/vikramg1/phanim/DFT_FETesting/buildDevelop/release/real/dftfe parameterFile_a.prm>CO_RefRun
srun /scratch/vikramg_root/vikramg1/phanim/DFT_FETesting/buildpCOHP/release/real/dftfe parameterFile_a.prm>H2_testCase_DFTFEKohnShamOrbital_slater_PAWCoor_pCOHP
#srun /scratch/vikramg_root/vikramg1/phanim/DFT_FETesting/buildSmearedChargeNew/release/real/dftfe fccAl_01Mod.prm>fccAl_GGA_GammaPoint_Orig_RealMode
#srun /scratch/vikramg_root/vikramg1/phanim/DFT_FETesting/buildStiffOptMerge/release/real/dftfe fccAl_01Mod.prm>fccAl_GGA_GammaPoint_Opt_RealMode_MoreBlocks


#ddt --offline -o n2debugExt.html /home/phanim/dftfeSmearedFloat/build/release/real/dftfe parameterFileNonPer.prm>smearedChargemassMatScalingFalseExtIntHangNode
#ddt --offline -o n2debugMPI.html -np 36 /home/phanim/dftfeSmearedFloat/build/release/real/dftfe parameterFileNonPer.prm>smearedChargemassMatScalingFalseExtIntA_MPI
#ddt --offline -o periodicUnitCellNew.html /scratch/vikramg_root/vikramg1/phanim/DFT_FETesting/buildDebug/release/complex/dftfe parameterFilePer.prm>debugTestAlUnitCellRun_LDAMemOpt_1procA

