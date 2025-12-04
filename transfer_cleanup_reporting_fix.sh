#!/bin/bash
# Transfer cleanup reporting fix

echo "Transferring cleanup reporting fix to cluster..."
echo "Fixes include:"
echo "1. ✅ Fixed cleanup for cached datasets (was being skipped)"
echo "2. ✅ Added proper error handling for cleanup operations"
echo "3. ✅ Improved reporting with clear success/failure indicators"
echo "4. ✅ Added emojis for better log readability"
echo "5. ✅ Fixed false reporting of cleanup when it didn't happen"
echo ""

echo "Transferring file..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO TEST CLEANUP REPORTING!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Expected improvements:"
echo "- ✅ Accurate cleanup reporting (no more false positives)"
echo "- ✅ Cleanup will work for both new and cached datasets"
echo "- ✅ Clear success/failure indicators in logs"
echo "- ✅ Better error handling for cleanup operations"
echo "- ✅ Honest reporting of what actually happened"
