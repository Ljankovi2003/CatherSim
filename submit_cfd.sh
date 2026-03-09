#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH -J "CatherCFD"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load cuda/12.2.1-gcc-11.3.1-sdqrj2e julia/1.10.8
export JULIA_NUM_THREADS=auto

# Run the CFD generation
julia --project=. generate_flow.jl
