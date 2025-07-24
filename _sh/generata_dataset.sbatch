#!/bin/bash
# SBATCH --job-name=mq-oracle # Job name
#SBATCH --chdir=/home/htc/khiroshi/SCRATCH/mq-oracle  # Navigate to the working directory where your script lies
#SBATCH --output=/home/htc/khiroshi/SCRATCH/mq-oracle/logs/%j.log     # Standard output and error log
#SBATCH --partition=small # Specify the desired partition, e.g. gpu or big. Note that small is the default partition.
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00  # set the maximum execution time to 3 days

echo 'Getting node information'
date;hostname;id;pwd

echo 'Activating virtual environment'
source ~/.profile  # Load the .profile
conda activate mq-oracle-exp
which python

echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Running python script'
python scripts/sagemath/polynomial_problem_generation.py