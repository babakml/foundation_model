#!/bin/bash
# Transfer download logic fix for downloaded but unprocessed files

echo "Transferring download logic fix to cluster..."
echo "Fix: Handle downloaded but unprocessed files correctly"
echo ""
echo "Problem: Pipeline was skipping 18 downloaded but unprocessed datasets"
echo "Solution: Added is_downloaded_but_not_processed() method"
echo ""

echo "Transferring file..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO PROCESS ALL REMAINING DATASETS!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Expected improvements:"
echo "- ✅ Will process 18 downloaded but unprocessed datasets"
echo "- ✅ Will download and process remaining ~196 datasets"
echo "- ✅ Total: ~214 datasets to process (vs skipping 18 before)"
echo "- ✅ Much better success rate expected!"
