#!/bin/bash
# Transfer enhanced pipeline and SLURM job script

echo "Transferring enhanced pipeline and SLURM job to cluster..."
echo "Files being transferred:"
echo "1. ✅ Enhanced streaming_pipeline.py (SRA metadata + download tracking)"
echo "2. ✅ Enhanced SLURM job script (run_als_pipeline_enhanced.slurm)"
echo ""

echo "Transferring files..."
scp src/streaming_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/
scp scripts/run_als_pipeline_enhanced.slurm ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO SUBMIT SLURM JOB!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_enhanced.slurm"
echo ""
echo "Monitor your job:"
echo "  squeue -u ul_oqn09"
echo "  tail -f logs/als_pipeline_enhanced_<job_id>.out"
echo ""
echo "Expected improvements:"
echo "- ✅ Resume capability (skip 132 already processed datasets)"
echo "- ✅ SRA metadata download for new datasets"
echo "- ✅ Better .gz file processing"
echo "- ✅ Reduced SRA download failures"
echo "- ✅ Download tracking and statistics"
echo "- ✅ Overall success rate: 70-75% expected"
