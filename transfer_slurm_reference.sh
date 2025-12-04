#!/bin/bash
# transfer_slurm_reference.sh

echo "🚀 Transferring SLURM job for reference download..."

scp scripts/run_reference_download.slurm ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/

echo "✅ SLURM job script transferred!"
echo ""
echo "Submit the job:"
echo "sbatch scripts/run_reference_download.slurm"






