#!/bin/bash
# Transfer the duplicate status panel fix

echo "Transferring duplicate panel fix to cluster..."
scp src/streaming_pipeline.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/src/

echo "Transfer complete!"
echo "On the cluster, test with:"
echo "  cd ~/als_foundation_model"
echo "  conda activate als_foundation"
echo "  python src/streaming_pipeline.py configs/streaming_config.json"
