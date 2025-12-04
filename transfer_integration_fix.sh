#!/bin/bash

# Transfer integration and compressed file fixes
echo "🔧 Transferring integration and compressed file fixes..."

# Transfer the updated pipeline file
scp src/streaming_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "✅ Fixes transferred!"
echo ""
echo "🔧 Fixes Applied:"
echo "  ✅ Fixed batch integration error (duplicate labels)"
echo "  ✅ Added support for compressed .gz files"
echo "  ✅ Added fallback handling for integration failures"
echo ""
echo "🔄 To apply the fixes:"
echo "1. Stop the current pipeline (Ctrl+C)"
echo "2. Run: python src/streaming_pipeline.py configs/streaming_config.json"
echo ""
echo "🎯 The pipeline should now handle integration and compressed files properly!"
