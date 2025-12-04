#!/bin/bash
# transfer_reference_scripts.sh

echo "🚀 Transferring reference genome and alignment scripts..."

scp scripts/download_references_and_indices.sh ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/
scp scripts/align_fastq_and_cleanup.sh ul_oqn09@uc3.scc.kit.edu:~/als_foundation_model/scripts/

echo "✅ Reference scripts transferred!"
echo ""
echo "Next steps on the cluster:"
echo "1. Download references and build STAR indices:"
echo "   bash scripts/download_references_and_indices.sh"
echo ""
echo "2. Align FASTQ files and remove on success:"
echo "   bash scripts/align_fastq_and_cleanup.sh ~/als_foundation_model/references/homo_sapiens/GRCh38_STAR 32"






