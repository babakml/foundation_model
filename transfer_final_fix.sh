#!/bin/bash
# transfer_final_fix.sh

echo "🚀 Transferring ALL fixes to cluster..."

# Transfer the fixed SRA downloader
scp src/sra_download_fix.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

# Transfer the fixed pipeline (with PosixPath fixes)  
scp src/parallel_optimized_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "✅ All fixes transferred!"
echo ""
echo "Next step - run the pipeline:"
echo "ssh ul_oqn09@uc3.scc.kit.edu 'cd ~/als_foundation_model && sbatch scripts/run_parallel_optimized_pipeline.slurm'"
