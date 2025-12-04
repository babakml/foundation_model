#!/bin/bash
# Transfer fixes to cluster - run this from your local machine

echo "🚀 Transferring SRA and PosixPath fixes to cluster..."

# Check if we're in the right directory
if [ ! -f "src/parallel_optimized_pipeline.py" ]; then
    echo "❌ Error: parallel_optimized_pipeline.py not found. Make sure you're in the project directory."
    exit 1
fi

if [ ! -f "src/sra_download_fix.py" ]; then
    echo "❌ Error: sra_download_fix.py not found. Make sure you're in the project directory."
    exit 1
fi

echo "📁 Found updated files, transferring..."

# Transfer the updated pipeline
echo "📤 Transferring parallel_optimized_pipeline.py..."
scp src/parallel_optimized_pipeline.py ul_oqn09@uc3n990:~/als_foundation_model/src/
if [ $? -eq 0 ]; then
    echo "✅ parallel_optimized_pipeline.py transferred successfully"
else
    echo "❌ Failed to transfer parallel_optimized_pipeline.py"
    exit 1
fi

# Transfer the updated SRA downloader
echo "📤 Transferring sra_download_fix.py..."
scp src/sra_download_fix.py ul_oqn09@uc3n990:~/als_foundation_model/src/
if [ $? -eq 0 ]; then
    echo "✅ sra_download_fix.py transferred successfully"
else
    echo "❌ Failed to transfer sra_download_fix.py"
    exit 1
fi

echo ""
echo "🎉 All files transferred successfully!"
echo ""
echo "Next steps on the cluster:"
echo "1. ssh ul_oqn09@uc3n990"
echo "2. cd ~/als_foundation_model"
echo "3. sbatch scripts/run_parallel_optimized_pipeline.slurm"
echo "4. Check status: squeue -u ul_oqn09"
echo "5. Monitor logs: tail -f logs/als_parallel_opt_*.out"
echo ""
echo "The fixes address:"
echo "- SRA rate limiting and SRR accession resolution"
echo "- PosixPath 'lower()' attribute errors"
echo "- 10X directory loading with string paths"
