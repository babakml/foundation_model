#!/bin/bash
#SBATCH --job-name=align_human
#SBATCH --output=logs/align_human_%j.out
#SBATCH --error=logs/align_human_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --account=ul

# Load required modules
echo "🔧 Loading modules and environment..."
module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate als_foundation

# Set working directory
cd ~/als_foundation_model
mkdir -p logs

# Paths
GENOME_DIR="~/als_foundation_model/references/homo_sapiens/GRCh38_STAR"
THREADS=${THREADS:-32}

# Print job information
echo "=========================================="
echo "Align Human FASTQs (STAR)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GenomeDir: $GENOME_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Run alignment and cleanup
echo "🧬 Running alignment and FASTQ cleanup (human)..."
bash scripts/align_fastq_and_cleanup.sh "$GENOME_DIR" "$THREADS"

# Completion
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="






