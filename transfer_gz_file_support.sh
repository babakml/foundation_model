#!/bin/bash
# Transfer .gz file support for CSV/TXT/TSV files

echo "Transferring .gz file support to cluster..."
echo "New features include:"
echo "1. ✅ Support for .csv.gz, .txt.gz, .tsv.gz files"
echo "2. ✅ Enhanced file patterns to include .gz variants"
echo "3. ✅ Improved compressed file reading for CSV/TXT matrices"
echo "4. ✅ Better download strategy for .gz files from GEO"
echo "5. ✅ Automatic detection and processing of compressed matrices"
echo ""

echo "Transferring file..."
scp src/streaming_pipeline_memory_optimized.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo ""
echo "🚀 READY TO PROCESS COMPRESSED CSV/TXT FILES!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  sbatch scripts/run_als_pipeline_memory_optimized.slurm"
echo ""
echo "Expected improvements:"
echo "- ✅ Will process GSE130938, GSE138120, GSE138121, GSE180122, GSE206330"
echo "- ✅ Support for .csv.gz, .txt.gz, .tsv.gz files from GEO"
echo "- ✅ Better success rate for text-based matrix data"
echo "- ✅ Handles compressed files that are common on GEO"
echo "- ✅ Much higher success rate for CSV/TXT datasets!"
