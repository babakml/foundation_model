#!/bin/bash
#SBATCH --job-name=ref_download
#SBATCH --output=logs/ref_download_%j.out
#SBATCH --error=logs/ref_download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --account=ul

# Load required modules
module load devel/miniforge/25.3.1-python-3.12

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate als_foundation

# Set working directory
cd ~/als_foundation_model

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export THREADS=32
export OMP_NUM_THREADS=32

# Print job information
echo "=========================================="
echo "Reference Genome Download and Index Building"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Install STAR if not already installed
echo "🔧 Installing STAR..."
conda install -y -c bioconda star

# Run the reference download and index building script
echo "🧬 Starting reference download and index building..."
bash scripts/download_references_and_indices.sh

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="

# Show disk usage
echo "📊 Disk usage after reference download:"
du -sh references/






