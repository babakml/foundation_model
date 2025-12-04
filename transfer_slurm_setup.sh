#!/bin/bash
# Transfer SLURM job script to cluster

echo "Transferring SLURM job script to cluster..."
scp scripts/run_als_pipeline.slurm ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/

echo "Transfer complete!"
echo ""
echo "On the cluster, you can now:"
echo "1. Submit the job: sbatch scripts/run_als_pipeline.slurm"
echo "2. Monitor the job: squeue -u ul_oqn09"
echo "3. Check logs: tail -f logs/als_pipeline_<job_id>.out"
echo "4. Cancel if needed: scancel <job_id>"
