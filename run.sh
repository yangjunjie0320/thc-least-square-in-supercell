#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=phi
#SBATCH --mem=500000

source /home/yangjunjie/.bashrc
export OMP_NUM_THREADS=28 # $SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=28 # $SLURM_CPUS_PER_TASK;
export PYSCF_MAX_MEMORY=50000 # $SLURM_MEM_PER_NODE;
export PYTHONUNBUFFERED=TRUE;
echo $PYSCF_MAX_MEMORY

conda activate py310-pyscf
export PYTHONPATH=/home/yangjunjie/packages/pyscf/py310-pyscf/:$PYTHONPATH
export TMPDIR=/scratch/global/yangjunjie/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
mkdir -p $TMPDIR
python phi-in-supercell.py

