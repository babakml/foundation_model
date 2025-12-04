#!/bin/bash
# transfer_storage_analysis.sh

echo "🚀 Transferring storage analysis script..."

scp analyze_storage_cleanup.py ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/

echo "✅ Storage analysis script transferred!"
echo ""
echo "Next step - run the analysis:"
echo "ssh ul_oqn09@uc3.scc.kit.edu 'cd ~/als_foundation_model && python analyze_storage_cleanup.py'"


