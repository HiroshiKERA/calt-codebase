#!/bin/bash
#SBATCH --job-name=calt-codebase # Job name
#SBATCH --chdir=/home/htc/khiroshi/SCRATCH/calt-dev/calt-codebase  # Navigate to the working directory where your script lies
#SBATCH --output=/home/htc/khiroshi/SCRATCH/calt-dev/logs/%j.log     # Standard output and error log
#SBATCH --partition=gpu # Specify the desired partition, e.g. gpu or big. Note that small is the default partition.
##SBATCH --nodelist=htc-gpu[001-004]
#SBATCH --gres=gpu:1  # The number of GPUs per task
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00  # set the maximum execution time to 3 days

echo 'Getting node information'
date;hostname;id;pwd

echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Running python script'
python scripts/sagemath/train.py --config config/train_partial_sum.yaml --dryrun