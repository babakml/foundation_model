#!/bin/bash
# Quick memory fix - reduce from 512GB to 256GB

echo "Quick fix: Reducing memory from 512GB to 256GB..."
scp scripts/run_parallel_optimized_pipeline.slurm ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/
echo "✅ Fixed! Memory reduced to 256GB"
echo ""
echo "Now you can run:"
echo "  sbatch scripts/run_parallel_optimized_pipeline.slurm"
