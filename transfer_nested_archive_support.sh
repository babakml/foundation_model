#!/bin/bash
# Transfer nested archive support for 10X Genomics data

echo "Transferring nested archive support to cluster..."
echo "New features include:"
echo "1. ✅ Automatic extraction of .tar.gz files inside tar files"
echo "2. ✅ Detection of 10X Genomics directories in nested structure"
echo "3. ✅ Support for GSE275999-style datasets (tar → tar.gz → 10X folders)"
echo "4. ✅ Improved file discovery for nested archives"
echo "5. ✅ Better handling of complex dataset structures"
echo ""

echo "Transferring file..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO PROCESS NESTED ARCHIVES!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Expected improvements:"
echo "- ✅ Will process GSE275999 and similar datasets"
echo "- ✅ Automatic extraction of nested .tar.gz files"
echo "- ✅ Detection of 10X matrix files in nested folders"
echo "- ✅ Much higher success rate for complex datasets"
echo "- ✅ Better handling of real-world data structures"
