#!/bin/bash
# Transfer SRA improvements and parallel processing to cluster

echo "Transferring SRA improvements and parallel processing to cluster..."
echo "Improvements include:"
echo "1. ✅ Fixed .gz compressed file processing"
echo "2. ✅ Improved SRA download logic with NCBI SRA database integration"
echo "3. ✅ Added multiple download strategies with retry logic"
echo "4. ✅ Created parallel processing pipeline"
echo "5. ✅ Better error handling and reporting"
echo ""

echo "Transferring files..."
scp src/streaming_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/
scp src/parallel_streaming_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "On the cluster, you can now:"
echo ""
echo "OPTION 1 - Test improved single-threaded pipeline:"
echo "  python src/streaming_pipeline.py configs/streaming_config.json"
echo ""
echo "OPTION 2 - Run parallel processing pipeline:"
echo "  python src/parallel_streaming_pipeline.py configs/streaming_config.json"
echo ""
echo "OPTION 3 - Submit parallel job to SLURM:"
echo "  # Edit run_als_pipeline.slurm to use parallel_streaming_pipeline.py"
echo "  sbatch scripts/run_als_pipeline.slurm"
echo ""
echo "Expected improvements:"
echo "- Better .gz file processing (should handle matrix.mtx.gz files)"
echo "- Reduced SRA download failures (uses NCBI SRA database validation)"
echo "- Parallel processing (2 download + 4 processing threads)"
echo "- More robust download with retry logic"
echo "- Better success rate overall (60-70% expected)"
