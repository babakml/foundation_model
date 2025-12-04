#!/bin/bash

# Quick fix for duplicate status panels
echo "🔧 Transferring duplicate status fix..."

# Transfer just the updated pipeline file
scp src/streaming_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "✅ Fix transferred!"
echo ""
echo "🔄 To apply the fix:"
echo "1. Stop the current pipeline (Ctrl+C)"
echo "2. Run: python src/streaming_pipeline.py configs/streaming_config.json"
echo ""
echo "🎯 This will eliminate the duplicate status panels!"
