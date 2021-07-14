#!/bin/bash
#SBATCH --job-name=one      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
##SBATCH --partition=hpg2-dev
#SBATCH --nodes=1 
##SBATCH --nodelist=c44a-s21
##SBATCH --exclude=dev1,dev2
#SBATCH --mem-per-cpu=800mb          # Memory per processor
#SBATCH --time=12:00:00              # Time limit hrs:min:sec
#SBATCH --output=meld.log     # Standard output and error log
##SBATCH --qos=alberto.perezant-b


unset PYTHONPATH
source ~/.load_OpenMM_cuda10_dev
#source /home/alberto.perezant/.load_MeldV2
[[ -d Data ]] || python setup_heuristics.py


if [ -e remd.log ]; then             #If there is a remd.log we are conitnuing a killed simulation
    prepare_restart --prepare-run  #so we need to prepare_restart
      fi

srun --mpi=pmix_v3  launch_remd --debug  
