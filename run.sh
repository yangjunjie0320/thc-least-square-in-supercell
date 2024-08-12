#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --job-name=phi
#SBATCH --mem=500000

source /home/yangjunjie/.bashrc
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE;
export PYTHONUNBUFFERED=TRUE;
echo $PYSCF_MAX_MEMORY

conda activate py310-pyscf
export PYTHONPATH=/home/yangjunjie/packages/pyscf/py310-pyscf/:$PYTHONPATH
export TMPDIR=/scratch/global/yangjunjie/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
mkdir -p $TMPDIR

mpiexec -n 4 python phi-in-supercell.py

