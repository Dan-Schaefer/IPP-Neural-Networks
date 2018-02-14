#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j

# Initial working directory:
#SBATCH -D ./

# Job Name:
#SBATCH -J test_slurm

# Queue (Partition):
#SBATCH --partition=express

# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#
#SBATCH --mail-type=none
#SBATCH --mail-user=daniel.schaefer@ipp.mpg.de
#

# Wall clock limit:
#SBATCH --time=00:25:00

# module purge
module load anaconda/3/5.0.0
module load cuda
module load cudnn/6.0
module load tensorflow

python ./2018-02-07_7D_efiITG_GB_5HiddenLayer_RELU_Dropout_20Epochs.py
# Run the program:
# srun ./myprog > prog.out
