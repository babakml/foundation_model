#!/bin/bash
# Setup script for ALS Foundation Model environment on cluster

echo "Setting up ALS Foundation Model environment..."

# Load required modules
module load anaconda3
module load cuda/11.8

# Create conda environment
conda create -n als_foundation python=3.9 -y
source activate als_foundation

# Install core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scanpy pandas numpy scikit-learn matplotlib seaborn
pip install anndata h5py loompy
pip install transformers datasets accelerate
pip install wandb tensorboard
pip install psutil tqdm

# Install bioinformatics tools
conda install -c bioconda sra-toolkit -y
conda install -c bioconda cellranger -y

# Install additional utilities
pip install openpyxl xlsxwriter
pip install harmony-pytorch

# Create project directories
mkdir -p /scratch/$USER/als_foundation/{data,model,cache,outputs}
mkdir -p /scratch/$USER/als_foundation/data/{raw,processed,metadata}
mkdir -p /scratch/$USER/als_foundation/model/{checkpoints,embeddings,logs}

# Set permissions
chmod -R 755 /scratch/$USER/als_foundation

echo "Environment setup completed!"
echo "To activate: source activate als_foundation"
echo "To run pipeline: sbatch scripts/run_streaming_pipeline.sh"
