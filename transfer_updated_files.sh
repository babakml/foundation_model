#!/bin/bash

# Transfer Updated ALS Foundation Model Pipeline to KIT Cluster
# This script transfers all the fixed files to the cluster

echo "🚀 Transferring Updated ALS Foundation Model Pipeline to KIT Cluster..."

# Create the project directory on the cluster
echo "📁 Creating project directory..."
ssh ul_oqn09@uc3.scc.kit.edu "mkdir -p ~/als_foundation_model"

# Transfer the updated pipeline code
echo "📦 Transferring updated pipeline code..."
scp -r src/ ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

# Transfer the updated configuration files
echo "⚙️ Transferring updated configuration files..."
scp -r configs/ ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

# Transfer all scripts
echo "🔧 Transferring all scripts..."
scp -r scripts/ ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

# Transfer requirements and data files
echo "📋 Transferring requirements and data files..."
scp requirements.txt ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp data_list_full.csv ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

# Transfer documentation
echo "📚 Transferring documentation..."
scp README_streaming_pipeline.md ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp ALS_single_cell_databases_info.txt ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp streaming_pipeline_design.md ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

echo "✅ Transfer complete!"
echo ""
echo "🔧 Key Updates Transferred:"
echo "  ✅ Fixed quality control bug (pct_counts_mt error)"
echo "  ✅ Added FTP download support with HTTP fallback"
echo "  ✅ Consolidated repetitive status output"
echo "  ✅ Fixed storage configuration (50TB limit)"
echo "  ✅ Improved file discovery patterns"
echo ""
echo "📋 Next Steps:"
echo "1. SSH to the cluster: ssh ul_oqn09@uc3.scc.kit.edu"
echo "2. Navigate to: cd ~/als_foundation_model"
echo "3. Activate environment: conda activate als_foundation"
echo "4. Test the pipeline: python src/streaming_pipeline.py configs/streaming_config.json"
echo ""
echo "🎯 The pipeline should now work much better with:"
echo "  • Faster FTP downloads"
echo "  • No processing failures"
echo "  • Cleaner output"
echo "  • Proper storage monitoring"
echo "  • Better file detection"
