#!/bin/bash
#SBATCH --job-name=als_foundation_enhanced
#SBATCH --output=logs/als_pipeline_enhanced_%j.out
#SBATCH --error=logs/als_pipeline_enhanced_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --account=ul
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
echo "ALS Foundation Model Pipeline Job (Enhanced)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Features: SRA metadata, download tracking, resume capability"
echo "=========================================="

# Check if download tracker exists
if [ -f "download_tracker.json" ]; then
    echo "Found existing download tracker - will resume from previous run"
    echo "Processed datasets will be skipped automatically"
else
    echo "No existing download tracker - starting fresh run"
fi

# Run the enhanced pipeline
python src/streaming_pipeline.py configs/streaming_config.json

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
print(f'Success rate: {(processed/total*100):.1f}%' if total > 0 else 'Success rate: 0.0%')
"
fi
