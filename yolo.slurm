#!/bin/bash
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=2048       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=yolov8test
#SBATCH --partition=gpu 
#SBATCH --gres=gpu 
#SBATCH --constraint='gpu_16gb&gpu_v100'
#SBATCH --error=/work/cseos2g/gclark/ai-football-chartmaker/job.%J.err
#SBATCH --output=/work/cseos2g/gclark/ai-football-chartmaker/job.%J.out


cd /work/cseos2g/gclark/ai-football-chartmaker

module load /common/cseos2g/gclark/yolov8

python yolov8Example.py
