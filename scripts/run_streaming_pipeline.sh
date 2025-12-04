#!/bin/bash
#SBATCH --job-name=als_streaming_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=streaming_pipeline_%j.out
#SBATCH --error=streaming_pipeline_%j.err

# Load required modules
module load anaconda3
module load cuda/11.8

# Activate conda environment
source activate als_foundation

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p /scratch/$USER/als_foundation/{data,model,cache,outputs}
mkdir -p /scratch/$USER/als_foundation/data/{raw,processed,metadata}
mkdir -p /scratch/$USER/als_foundation/model/{checkpoints,embeddings,logs}

# Change to project directory
cd /home/$USER/foundation_model

# Run the streaming pipeline
python src/streaming_pipeline.py configs/streaming_config.json

# Check exit status
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully"
else
    echo "Pipeline failed with exit code $?"
    exit 1
fi

# Clean up temporary files
rm -rf /scratch/$USER/als_foundation/cache/*

echo "Job completed at $(date)"
