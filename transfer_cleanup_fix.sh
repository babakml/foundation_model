#!/bin/bash
# Transfer cleanup fix for raw data removal

echo "Transferring cleanup fix to cluster..."
echo "Fix: Corrected raw data cleanup logic to use proper config setting"
echo ""
echo "Problem: Pipeline wasn't removing raw data after processing"
echo "Solution: Fixed config key mismatch (cleanup_raw_data vs keep_raw_data)"
echo ""

echo "Transferring file..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO RUN WITH PROPER CLEANUP!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Expected improvements:"
echo "- ✅ Raw data will be properly removed after processing"
echo "- ✅ Storage space will be freed up automatically"
echo "- ✅ Only processed H5AD files will be kept"
echo "- ✅ Much better storage management!"
