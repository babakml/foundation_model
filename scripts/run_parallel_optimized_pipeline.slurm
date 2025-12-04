#!/bin/bash
#SBATCH --job-name=als_parallel_optimized
#SBATCH --output=logs/als_parallel_opt_%j.out
#SBATCH --error=logs/als_parallel_opt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --account=ul

# Load required modules
module load devel/miniforge/25.3.1-python-3.12

# Activate conda environment
source activate als_foundation

# Set working directory
cd ~/als_foundation_model

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for better performance
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_MAX_THREADS=64

# Print job information
echo "=========================================="
echo "ALS Foundation Model - Parallel Optimized Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB (256GB)"
echo "Start time: $(date)"
echo "Features:"
echo "  - Parallel download and processing"
echo "  - Enhanced nested archive support"
echo "  - Optimized memory usage (256GB)"
echo "  - 64 CPU cores"
echo "  - FTP downloads for speed"
echo "  - No email addresses"
echo "=========================================="

# Check if download tracker exists
if [ -f "download_tracker.json" ]; then
    echo "Found existing download tracker - will resume from previous run"
    echo "Processed datasets will be skipped automatically"
    echo "Downloaded but unprocessed datasets will be processed"
else
    echo "No existing download tracker - starting fresh run"
fi

# Run the parallel optimized pipeline
python src/parallel_optimized_pipeline.py configs/streaming_config.json

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="

# Show download tracker statistics
if [ -f "download_tracker.json" ]; then
    echo ""
    echo "Download Tracker Summary:"
    echo "========================="
    python -c "
import json
with open('download_tracker.json', 'r') as f:
    data = json.load(f)
total = len(data)
downloaded = sum(1 for d in data.values() if d.get('download_complete', False))
processed = sum(1 for d in data.values() if d.get('processing_complete', False))
failed = sum(1 for d in data.values() if d.get('failed', False))
print(f'Total datasets: {total}')
print(f'Downloaded: {downloaded}')
print(f'Processed: {processed}')
print(f'Failed: {failed}')
print(f'Download success rate: {(downloaded/total*100):.1f}%' if total > 0 else 'Download success rate: 0.0%')
print(f'Processing success rate: {(processed/total*100):.1f}%' if total > 0 else 'Processing success rate: 0.0%')
"
fi
