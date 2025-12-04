#!/bin/bash
# Fixed SRA Toolkit installation script
# More robust error handling and debugging

echo "🔧 Installing SRA Toolkit on the cluster (Fixed Version)..."
echo "============================================================="

# Set installation directory
INSTALL_DIR="$HOME/sra_toolkit"
DOWNLOAD_FILE="sratoolkit.current-centos_linux64.tar.gz"

echo "📥 Downloading SRA Toolkit..."

# Download SRA Toolkit with better error handling
cd $HOME

# Try wget first
echo "Trying wget..."
wget -q --timeout=30 "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-centos_linux64.tar.gz"

if [ $? -ne 0 ]; then
    echo "wget failed, trying curl..."
    rm -f "$DOWNLOAD_FILE"
    curl -L --connect-timeout 30 -o "$DOWNLOAD_FILE" "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-centos_linux64.tar.gz"
    
    if [ $? -ne 0 ]; then
        echo "❌ Both wget and curl failed!"
        echo "Trying alternative download method..."
        
        # Try downloading a specific version
        rm -f "$DOWNLOAD_FILE"
        wget -q "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.0.11/sratoolkit.3.0.11-centos_linux64.tar.gz"
        
        if [ $? -ne 0 ]; then
            echo "❌ All download methods failed!"
            echo "Please check your internet connection and try again."
            exit 1
        fi
    fi
fi

# Check if file was downloaded
if [ ! -f "$DOWNLOAD_FILE" ]; then
    echo "❌ Download file not found!"
    exit 1
fi

echo "✅ Download successful"
echo "📁 File size: $(ls -lh $DOWNLOAD_FILE | awk '{print $5}')"

echo "📦 Extracting SRA Toolkit..."

# Check if file is valid tar.gz
file "$DOWNLOAD_FILE"
echo "File type: $(file $DOWNLOAD_FILE)"

# Try extraction with verbose output
tar -tzf "$DOWNLOAD_FILE" | head -5
echo "Archive contents preview:"
tar -tzf "$DOWNLOAD_FILE" | head -5

# Extract with verbose output
echo "Extracting with verbose output..."
tar -xzf "$DOWNLOAD_FILE" -v

if [ $? -eq 0 ]; then
    echo "✅ Extraction successful"
    
    # Find the extracted directory
    TOOLKIT_DIR=$(ls -d sratoolkit* 2>/dev/null | head -1)
    
    if [ -z "$TOOLKIT_DIR" ]; then
        echo "❌ No sratoolkit directory found after extraction!"
        echo "Contents of current directory:"
        ls -la
        exit 1
    fi
    
    echo "📁 Found toolkit directory: $TOOLKIT_DIR"
    
    # Check if it's a directory
    if [ ! -d "$TOOLKIT_DIR" ]; then
        echo "❌ $TOOLKIT_DIR is not a directory!"
        exit 1
    fi
    
    # Check if it contains the expected files
    if [ ! -f "$TOOLKIT_DIR/bin/prefetch" ]; then
        echo "❌ prefetch binary not found in $TOOLKIT_DIR/bin/"
        echo "Contents of $TOOLKIT_DIR/bin/:"
        ls -la "$TOOLKIT_DIR/bin/" | head -10
        exit 1
    fi
    
    echo "✅ Toolkit directory structure looks correct"
    
    # Move to installation directory
    echo "📁 Moving to installation directory: $INSTALL_DIR"
    rm -rf "$INSTALL_DIR"
    mv "$TOOLKIT_DIR" "$INSTALL_DIR"
    
    if [ ! -d "$INSTALL_DIR" ]; then
        echo "❌ Failed to move toolkit to $INSTALL_DIR"
        exit 1
    fi
    
    echo "✅ Toolkit moved to $INSTALL_DIR"
    
    # Configure SRA Toolkit
    echo "⚙️ Configuring SRA Toolkit..."
    cd "$INSTALL_DIR"
    
    # Create ncbi directory
    mkdir -p "$HOME/ncbi/public"
    
    # Configure SRA Toolkit (non-interactive)
    echo "Setting up configuration..."
    ./bin/vdb-config --set /repository/user/main/public/root="$HOME/ncbi/public" 2>/dev/null || true
    ./bin/vdb-config --set /repository/user/main/public/apps/file/volumes/sraPileup="$HOME/ncbi/public/sraPileup" 2>/dev/null || true
    ./bin/vdb-config --set /repository/user/main/public/apps/refseq/volumes/refseq="$HOME/ncbi/public/refseq" 2>/dev/null || true
    ./bin/vdb-config --set /repository/user/main/public/apps/wgs/volumes/wgs="$HOME/ncbi/public/wgs" 2>/dev/null || true
    
    echo "✅ Configuration complete"
    
    # Test installation
    echo "🧪 Testing SRA Toolkit installation..."
    
    echo "Testing prefetch..."
    ./bin/prefetch --version
    PREFETCH_STATUS=$?
    
    echo "Testing fasterq-dump..."
    ./bin/fasterq-dump --version
    FASTERQ_STATUS=$?
    
    echo "Testing vdb-config..."
    ./bin/vdb-config --version
    VDB_STATUS=$?
    
    if [ $PREFETCH_STATUS -eq 0 ] && [ $FASTERQ_STATUS -eq 0 ] && [ $VDB_STATUS -eq 0 ]; then
        echo "✅ All SRA Toolkit components working!"
        
        # Add to PATH
        echo "📝 Adding SRA Toolkit to PATH..."
        echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> ~/.bashrc
        echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> ~/.bash_profile
        
        # Create symlinks for easy access
        mkdir -p ~/bin
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
        echo "Prefetch status: $PREFETCH_STATUS"
        echo "Fasterq-dump status: $FASTERQ_STATUS"
        echo "VDB-config status: $VDB_STATUS"
        exit 1
    fi
    
else
    echo "❌ Extraction failed!"
    echo "Trying alternative extraction method..."
    
    # Try with different tar options
    tar -xzf "$DOWNLOAD_FILE" --no-same-owner
    
    if [ $? -eq 0 ]; then
        echo "✅ Alternative extraction successful"
        # Continue with the rest of the installation
    else
        echo "❌ All extraction methods failed!"
        echo "Please check the downloaded file:"
        ls -la "$DOWNLOAD_FILE"
        file "$DOWNLOAD_FILE"
        exit 1
    fi
fi

# Clean up
echo "🧹 Cleaning up..."
rm -f "$HOME/$DOWNLOAD_FILE"

echo "✅ Installation script completed!"
