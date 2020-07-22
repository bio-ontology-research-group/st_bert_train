#!/bin/bash --login

#SBATCH --time 4-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --constraint=gtx1080ti
#SBATCH --partition=batch
#SBATCH -e slurm.%N.%j.err # STDERR 


# activate the conda environment
conda activate /ibex/scratch/kafkass/NER/conda-environment-examples/env

# start the nvdashboard server in the background
NVDASHBOARD_PORT=8000
python -m jupyterlab_nvdashboard.server $NVDASHBOARD_PORT &
NVDASHBOARD_PID=$!

# run the training script
python /ibex/scratch/kafkass/NER/simple_transformers/train_cuda_$1.py >/ibex/scratch/kafkass/NER/simple_transformers/cuda_$1_outputs/$1_results.txt

# kill off the GPU monitoring processes
kill $NVDASHBOARD_PID
