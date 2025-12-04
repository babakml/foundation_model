#!/bin/bash
#SBATCH --job-name=als_foundation
#SBATCH --output=logs/als_pipeline_%j.out
#SBATCH --error=logs/als_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=default
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ul_oqn09@kit.edu

# Load required modules
module load devel/miniforge/25.3.1-python-3.12

# Activate conda environment
source activate als_foundation

# Set working directory
cd ~/als_foundation_model

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for better performance
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

# Print job information
echo "=========================================="
echo "ALS Foundation Model Pipeline Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Run the pipeline
python src/streaming_pipeline.py configs/streaming_config.json

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="
