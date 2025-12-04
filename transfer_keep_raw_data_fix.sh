#!/bin/bash
# Transfer keep_raw_data configuration fix

echo "Transferring keep_raw_data configuration fix to cluster..."
echo "Changes made:"
echo "1. ✅ Set keep_raw_data: true (as requested)"
echo "2. ✅ Pipeline will now KEEP all raw files"
echo "3. ✅ No cleanup of raw data will occur"
echo "4. ✅ Raw files will be preserved for manual inspection"
echo ""

echo "Transferring files..."
scp configs/streaming_config.json ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/configs/
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO RUN WITH RAW DATA PRESERVATION!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Expected behavior:"
echo "- 📁 All raw files will be KEPT (no cleanup)"
echo "- ✅ Accurate reporting: 'Keeping raw data for dataset_id (keep_raw_data=True)'"
echo "- 🔍 Raw files preserved for manual inspection of failed datasets"
echo "- 📊 You can analyze which datasets failed and why"
