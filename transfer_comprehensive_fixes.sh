#!/bin/bash
# Transfer comprehensive fixes for ALS pipeline

echo "Transferring comprehensive fixes to cluster..."
echo "Fixes include:"
echo "1. Fixed .gz compressed file processing"
echo "2. Improved SRA download logic with proper detection"
echo "3. Added multiple download strategies"
echo "4. Added retry logic with exponential backoff"
echo "5. Better error handling and reporting"
echo ""

scp src/streaming_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "On the cluster, you can now:"
echo "1. Test the fixes: python src/streaming_pipeline.py configs/streaming_config.json"
echo "2. Or submit a new job: sbatch scripts/run_als_pipeline.slurm"
echo ""
echo "Expected improvements:"
echo "- Better .gz file processing (should handle matrix.mtx.gz files)"
echo "- Reduced SRA download failures (only tries SRA when appropriate)"
echo "- More robust download with retry logic"
echo "- Better success rate overall"
