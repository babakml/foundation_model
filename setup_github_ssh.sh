#!/bin/bash
# Script to set up SSH authentication for GitHub

echo "🔐 Setting up SSH authentication for GitHub..."

# Check if SSH key exists
if [ ! -f ~/.ssh/id_ed25519 ] && [ ! -f ~/.ssh/id_rsa ]; then
    echo "📝 Generating SSH key..."
    ssh-keygen -t ed25519 -C "babak.loghmani@gmail.com" -f ~/.ssh/id_ed25519 -N ""
    echo "✅ SSH key generated"
else
    echo "✅ SSH key already exists"
fi

# Start ssh-agent
eval "$(ssh-agent -s)"

# Add SSH key to agent
if [ -f ~/.ssh/id_ed25519 ]; then
    ssh-add ~/.ssh/id_ed25519
elif [ -f ~/.ssh/id_rsa ]; then
    ssh-add ~/.ssh/id_rsa
fi

# Display public key
echo ""
echo "📋 Your public SSH key:"
echo "=========================================="
if [ -f ~/.ssh/id_ed25519.pub ]; then
    cat ~/.ssh/id_ed25519.pub
elif [ -f ~/.ssh/id_rsa.pub ]; then
    cat ~/.ssh/id_rsa.pub
fi
echo "=========================================="
echo ""
echo "📝 Next steps:"
echo "1. Copy the public key above"
echo "2. Go to: https://github.com/settings/keys"
echo "3. Click 'New SSH key'"
echo "4. Paste the key and save"
echo "5. Then update your remote URL:"
echo "   git remote set-url origin git@github.com:babakml/foundation_model.git"
echo "6. Test connection: ssh -T git@github.com"
echo "7. Push: git push -u origin main"

