#!/bin/bash
# Transfer method name fix for memory-optimized pipeline

echo "Transferring method name fix to cluster..."
echo "Fix: Corrected check_storage() to check_storage_space() method call"
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
echo "The method name error has been fixed - pipeline should now run properly!"
