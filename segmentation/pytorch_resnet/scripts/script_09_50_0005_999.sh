#!/bin/bash

#SBATCH --job-name train_resnet
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 64gb
#SBATCH --time 48:00:00
#SBATCH --constraint interconnect_hdr

export OMP_NUM_THREADS=8

module load anaconda3

source activate diss

cd /home/mfaykus/dissertation/segmentation/pytorch_resnet

python train.py 09 50 0005 999
