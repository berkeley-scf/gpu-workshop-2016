#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --partition=savio2_gpu
#SBATCH --account=ac_scsguest
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --mail-user=paciorek@stat.berkeley.edu

module load cuda
module unload intel  # do this to avoid compilation issues

# insert code here to run your computations
