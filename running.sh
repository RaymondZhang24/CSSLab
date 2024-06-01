#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=raymond
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4


conda activate KGE
bash run.sh train DistMult FB15k-237 0 2 1024 256 500 200.0 1.0 0.001 10000 16 -r 0.00001
