#! /bin/sh
#SBATCH -N 2
#SBATCH -p cnall
srun hostname
srun ./monitor.sh