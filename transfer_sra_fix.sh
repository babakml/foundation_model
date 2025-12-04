#!/bin/bash
# Transfer fixed SRA toolkit installation script

echo "Transferring fixed SRA toolkit installation script..."
scp scripts/install_sra_toolkit_fixed.sh ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/
echo "✅ Transfer complete!"
echo ""
echo "On the cluster, run:"
echo "  cd ~/als_foundation_model"
echo "  chmod +x scripts/install_sra_toolkit_fixed.sh"
echo "  ./scripts/install_sra_toolkit_fixed.sh"
echo ""
echo "This fixed version includes:"
echo "- ✅ Better error handling"
echo "- ✅ Verbose output for debugging"
echo "- ✅ Multiple download methods"
echo "- ✅ File validation checks"
echo "- ✅ Alternative extraction methods"
