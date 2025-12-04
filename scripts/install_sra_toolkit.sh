#!/bin/bash
# Install SRA Toolkit on the cluster
# Based on NCBI documentation: https://www.ncbi.nlm.nih.gov/sra/docs/sradownload/

echo "🔧 Installing SRA Toolkit on the cluster..."
echo "============================================="

# Set installation directory
INSTALL_DIR="$HOME/sra_toolkit"
TOOLKIT_VERSION="3.0.11"

echo "📥 Downloading SRA Toolkit version $TOOLKIT_VERSION..."

# Download SRA Toolkit
cd $HOME
wget -q "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-centos_linux64.tar.gz"

if [ $? -eq 0 ]; then
    echo "✅ Download successful"
else
    echo "❌ Download failed - trying alternative method"
    # Try alternative download method
    curl -L -o "sratoolkit.current-centos_linux64.tar.gz" "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-centos_linux64.tar.gz"
fi

echo "📦 Extracting SRA Toolkit..."
tar -xzf sratoolkit.current-centos_linux64.tar.gz

# Find the extracted directory
TOOLKIT_DIR=$(ls -d sratoolkit.current-centos_linux64* | head -1)

if [ -d "$TOOLKIT_DIR" ]; then
    echo "✅ Extraction successful"
    echo "📁 Toolkit directory: $TOOLKIT_DIR"
    
    # Move to installation directory
    mv "$TOOLKIT_DIR" "$INSTALL_DIR"
    
    echo "⚙️ Configuring SRA Toolkit..."
    cd "$INSTALL_DIR"
    
    # Configure SRA Toolkit (non-interactive)
    ./bin/vdb-config --set /repository/user/main/public/root="$HOME/ncbi/public"
    ./bin/vdb-config --set /repository/user/main/public/apps/file/volumes/sraPileup="$HOME/ncbi/public/sraPileup"
    ./bin/vdb-config --set /repository/user/main/public/apps/refseq/volumes/refseq="$HOME/ncbi/public/refseq"
    ./bin/vdb-config --set /repository/user/main/public/apps/wgs/volumes/wgs="$HOME/ncbi/public/wgs"
    
    echo "✅ Configuration complete"
    
    # Test installation
    echo "🧪 Testing SRA Toolkit installation..."
    ./bin/prefetch --version
    ./bin/fasterq-dump --version
    ./bin/vdb-config --version
    
    if [ $? -eq 0 ]; then
        echo "✅ SRA Toolkit installation successful!"
        
        # Add to PATH
        echo "📝 Adding SRA Toolkit to PATH..."
        echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> ~/.bashrc
        echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> ~/.bash_profile
        
        # Create symlinks for easy access
        ln -sf "$INSTALL_DIR/bin/prefetch" ~/bin/prefetch 2>/dev/null || true
        ln -sf "$INSTALL_DIR/bin/fasterq-dump" ~/bin/fasterq-dump 2>/dev/null || true
        ln -sf "$INSTALL_DIR/bin/vdb-config" ~/bin/vdb-config 2>/dev/null || true
        
        echo ""
        echo "🎉 SRA Toolkit installation complete!"
        echo "====================================="
        echo "Installation directory: $INSTALL_DIR"
        echo "Binaries available at: $INSTALL_DIR/bin"
        echo ""
        echo "To use SRA Toolkit in your current session:"
        echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\""
        echo ""
        echo "To test the installation:"
        echo "prefetch --version"
        echo "fasterq-dump --version"
        echo ""
        echo "The toolkit has been added to your ~/.bashrc for future sessions."
        
    else
        echo "❌ SRA Toolkit installation failed!"
        exit 1
    fi
    
else
    echo "❌ Extraction failed!"
    exit 1
fi

# Clean up
echo "🧹 Cleaning up..."
rm -f "$HOME/sratoolkit.current-centos_linux64.tar.gz"

echo "✅ Installation script completed!"
