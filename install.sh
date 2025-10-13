#!/bin/bash
set -e

echo "🎯 Installing Modular RAG System"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Python
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1-2 || echo "none")
if [[ "$python_version" == "none" ]]; then
    print_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi
print_success "Python $python_version found"

# Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip > /dev/null

# Install requirements
print_status "Installing requirements..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    print_success "Requirements installed"
else
    # Minimal install
    pip install requests numpy
    print_warning "No requirements.txt found, installed minimal dependencies"
fi

# Make scripts executable
print_status "Setting up permissions..."
chmod +x rag_launcher.py
chmod +x enhanced_rag_system.py
[ -f setup_rag_system.sh ] && chmod +x setup_rag_system.sh
print_success "Permissions set"

# Create storage directories
print_status "Creating storage structure..."
mkdir -p rag_storage/uploads/{json,txt}
print_success "Storage directories created"

# Test basic functionality
print_status "Testing installation..."
if python3 -c "from enhanced_rag_system import EnhancedRAGSystem; print('✅ Core modules working')" 2>/dev/null; then
    print_success "Installation test passed"
else
    print_warning "Some modules may need additional dependencies"
fi

echo ""
echo "🎉 Installation Complete!"
echo "========================"
print_success "✅ Virtual environment: venv/"
print_success "✅ Core dependencies installed"
print_success "✅ Storage structure created"
print_success "✅ Scripts are executable"

echo ""
echo "🚀 Next Steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Launch system: python3 rag_launcher.py"
echo "  3. Upload files to: rag_storage/uploads/"
echo "  4. Access web interface: http://localhost:8089"

echo ""
echo "📖 For advanced setup, run: ./setup_rag_system.sh"