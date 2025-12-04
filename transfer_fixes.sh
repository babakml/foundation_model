#!/bin/bash
# Transfer the fixes for SRA rate limiting and PosixPath bugs

echo "🚀 Transferring fixes to cluster..."

# Transfer the updated pipeline
scp src/parallel_optimized_pipeline.py ul_oqn09@uc3n990:~/als_foundation_model/src/

# Transfer the updated SRA downloader
scp src/sra_download_fix.py ul_oqn09@uc3n990:~/als_foundation_model/src/

echo "✅ Files transferred successfully!"
echo ""
echo "Next steps on the cluster:"
echo "1. cd ~/als_foundation_model"
echo "2. sbatch scripts/run_parallel_optimized_pipeline.slurm"
echo "3. Check status with: squeue -u ul_oqn09"
echo "4. Monitor logs with: tail -f logs/als_parallel_opt_*.out"
