#!/bin/bash

# Transfer script for ALS Foundation Model Pipeline to KIT Cluster
# Run this from your local machine after SSH'ing to the cluster

echo "🚀 Transferring ALS Foundation Model Pipeline to KIT Cluster..."

# Create the project directory on the cluster
echo "📁 Creating project directory..."
mkdir -p ~/als_foundation_model

# Transfer the main pipeline code
echo "📦 Transferring pipeline code..."
scp -r src/ ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp -r scripts/ ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp -r configs/ ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

# Transfer configuration files
echo "⚙️ Transferring configuration files..."
scp requirements.txt ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp data_list_full.csv ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp README_streaming_pipeline.md ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

# Transfer documentation
echo "📚 Transferring documentation..."
scp ALS_single_cell_databases_info.txt ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/
scp streaming_pipeline_design.md ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

echo "✅ Transfer complete!"
echo ""
echo "Next steps:"
echo "1. SSH to the cluster: ssh ul_oqn09@uc3.scc.kit.edu"
echo "2. Navigate to: cd ~/als_foundation_model"
echo "3. Run setup: bash scripts/setup_environment.sh"
echo "4. Configure paths in configs/streaming_config.json"
echo "5. Test the pipeline: python scripts/test_data_loading.py"
echo "6. Start the job: sbatch scripts/run_streaming_pipeline.sh"
