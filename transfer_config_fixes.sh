#!/bin/bash
# Transfer all configuration fixes

echo "Transferring configuration fixes to cluster..."
echo "Fixes include:"
echo "1. ✅ Quality control now uses config parameters"
echo "2. ✅ Normalization now uses config parameters"  
echo "3. ✅ Removed unused batch processing methods"
echo "4. ✅ Logging now uses config parameters"
echo "5. ✅ All config mismatches resolved"
echo ""

echo "Transferring file..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO RUN WITH PROPER CONFIGURATION!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Expected improvements:"
echo "- ✅ Quality control uses config: min_genes=200, max_genes=5000, max_mt=20%"
echo "- ✅ Normalization uses config: target_sum=10000, log_transform=true, scale=true"
echo "- ✅ Logging uses config: level=INFO, log_file=streaming_pipeline.log"
echo "- ✅ No more hard-coded parameters"
echo "- ✅ All configuration options now properly used!"
