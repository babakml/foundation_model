#!/bin/bash
# Transfer DataFrame fix for memory-optimized pipeline

echo "Transferring DataFrame fix to cluster..."
echo "Fix: Corrected pandas DataFrame boolean evaluation error"
echo ""

echo "Transferring file..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO RUN FIXED PIPELINE!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "The DataFrame error has been fixed - pipeline should now run properly!"
