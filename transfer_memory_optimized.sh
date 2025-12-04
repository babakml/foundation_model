#!/bin/bash
# Transfer memory-optimized pipeline and SLURM job script

echo "Transferring memory-optimized pipeline to cluster..."
echo "New features include:"
echo "1. ✅ Memory-optimized processing (one dataset at a time)"
echo "2. ✅ 256GB memory allocation"
echo "3. ✅ Aggressive memory cleanup and garbage collection"
echo "4. ✅ Individual dataset saving (no batching)"
echo "5. ✅ Real-time memory monitoring"
echo "6. ✅ All previous fixes (.gz files, SRA validation, retry logic)"
echo ""

echo "Transferring files..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/
scp scripts/run_als_pipeline_memory_optimized.slurm ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO SUBMIT MEMORY-OPTIMIZED SLURM JOB!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Monitor your job:"
echo "  squeue -u ul_oqn09"
echo "  tail -f logs/als_pipeline_memory_opt_<job_id>.out"
echo ""
echo "Expected improvements:"
echo "- ✅ No more OUT_OF_MEMORY errors (256GB RAM)"
echo "- ✅ Memory-optimized processing (one dataset at a time)"
echo "- ✅ Aggressive memory cleanup after each dataset"
echo "- ✅ Resume capability (skip already processed datasets)"
echo "- ✅ Real-time memory monitoring"
echo "- ✅ Overall success rate: 80-90% expected"
