#!/bin/bash
# Script to set up GitHub repository for foundation_model project

echo "🔧 Setting up GitHub repository..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not installed."
    echo "   Install it with: brew install gh"
    echo "   Or download from: https://cli.github.com/"
    exit 1
fi

# Login to GitHub (will prompt for credentials)
echo "📝 Logging in to GitHub..."
gh auth login

# Create the repository
echo "📦 Creating repository 'foundation_model'..."
gh repo create foundation_model --public --description "Foundation Model - Single-cell RNA-seq data processing pipeline"

# Initialize git if not already done
if [ ! -d .git ]; then
    echo "🔨 Initializing git repository..."
    git init
fi

# Add remote
echo "🔗 Adding remote repository..."
git remote add origin https://github.com/$(gh api user --jq .login)/foundation_model.git 2>/dev/null || \
git remote set-url origin https://github.com/$(gh api user --jq .login)/foundation_model.git

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Data files
data/raw/
data/processed/
data/synapse/
synapse_downloads/
*.h5ad
*.h5
*.fastq
*.fastq.gz
*.bam
*.bai

# Logs
logs/
*.log
*.out
*.err

# Config files with credentials
.synapseConfig
*.key
*.pem

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
*.bak
*.backup

# SLURM
*.slurm
slurm-*.out
slurm-*.err

# Jupyter
.ipynb_checkpoints/
*.ipynb

# References (too large)
references/

# Download tracker (may contain sensitive info)
download_tracker.json

# Transfer scripts (not needed on GitHub, may contain ALS references)
transfer_*.sh

# Old documentation files with specific references
README_streaming_pipeline.md
streaming_pipeline_design.md
ALS_single_cell_databases_info.txt
EOF
    echo "✅ Created .gitignore"
fi

echo ""
echo "✅ Repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and add files: git add ."
echo "2. Commit: git commit -m 'Initial commit'"
echo "3. Push: git push -u origin main"
echo ""
echo "Or if using master branch:"
echo "3. Push: git push -u origin master"
echo ""
echo "⚠️  Remember: Use a generic commit message like:"
echo "   git commit -m 'Initial commit: Foundation Model pipeline'"

