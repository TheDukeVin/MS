#!/bin/bash
#SBATCH -c 20 # Number of cores requested
#SBATCH -t 2000 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=100000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o MS_std.out # Standard out goes to this file
#SBATCH -e MS_std.err # Standard err goes to this filehostname
srun -l a.out