#!/bin/bash
set -e

echo "🎯 Setting up Integrated RAG System"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "run_integrated_rag.py" ]]; then
    print_error "Must be run from the RAG directory"
    print_error "Please cd to /media/nike/5f57e86a-891a-4785-b1c8-fae01ada4edd1/Modular Deepdive/RAG/"
    exit 1
fi

print_status "Checking Python environment..."

# Check Python version
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1-2 || echo "none")
if [[ "$python_version" == "none" ]]; then
    print_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi

print_success "Python $python_version found"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install base requirements
print_status "Installing base Python packages..."
pip install --quiet \
    requests \
    numpy \
    tkinter \
    pathlib \
    dataclasses \
    logging \
    typing

print_success "Base packages installed"

# Try to install optional but recommended packages
print_status "Installing optional packages..."

# ChromaDB for vector storage
if pip install --quiet chromadb 2>/dev/null; then
    print_success "✅ ChromaDB installed (recommended vector database)"
else
    print_warning "⚠️  ChromaDB not installed - will use SQLite fallback"
fi

# Sentence Transformers for embeddings
if pip install --quiet sentence-transformers 2>/dev/null; then
    print_success "✅ Sentence Transformers installed (for embedding generation)"
else
    print_warning "⚠️  Sentence Transformers not installed - will try Ollama or fallback"
fi

# FAISS for similarity search
if pip install --quiet faiss-cpu 2>/dev/null; then
    print_success "✅ FAISS installed (for fast similarity search)"
else
    print_warning "⚠️  FAISS not installed - will use slower similarity search"
fi

# tiktoken for token counting
if pip install --quiet tiktoken 2>/dev/null; then
    print_success "✅ tiktoken installed (for accurate token counting)"
else
    print_warning "⚠️  tiktoken not installed - will use word-based estimation"
fi

# PyQt5 for better GUI (optional)
if pip install --quiet PyQt5 2>/dev/null; then
    print_success "✅ PyQt5 installed (enhanced GUI support)"
else
    print_warning "⚠️  PyQt5 not installed - using basic Tkinter GUI"
fi

# System dependencies for keystroke logging
print_status "Checking system dependencies..."

# Check for pynput (keystroke logging)
if pip install --quiet pynput 2>/dev/null; then
    print_success "✅ pynput installed (keystroke logging)"
else
    print_warning "⚠️  pynput not available - keystroke logging may be limited"
fi

# Check for X11 dependencies (Linux)
if [[ -f /usr/include/X11/Xlib.h ]] || [[ -f /usr/X11R6/include/X11/Xlib.h ]]; then
    print_success "✅ X11 development headers found"
else
    print_warning "⚠️  X11 development headers not found"
    print_warning "    Install with: sudo apt-get install libx11-dev libxext-dev"
fi

# Create storage directories
print_status "Setting up storage directories..."
mkdir -p rag_storage/rag_data
mkdir -p rag_storage/keystrokes
mkdir -p rag_storage/ocr_data
mkdir -p rag_storage/embeddings
print_success "Storage directories created"

# Create default configuration
print_status "Creating default configuration..."
cat > rag_config.json << 'EOF'
{
  "storage_path": "./rag_storage",
  "keystroke_logging": {
    "enabled": true,
    "privacy_mode": true,
    "session_timeout": 600,
    "batch_size": 5
  },
  "embedding_pipeline": {
    "chunk_size": 512,
    "cache_size": 1000,
    "batch_timeout": 30
  },
  "ocr_integration": {
    "enabled": true,
    "analysis_interval": 10,
    "confidence_threshold": 0.7
  },
  "query_interface": {
    "gui_enabled": true,
    "background_processing": true
  },
  "rag_system": {
    "vector_backend": "auto",
    "max_documents": 10000,
    "similarity_threshold": 0.7
  }
}
EOF
print_success "Default configuration created: rag_config.json"

# Make scripts executable
print_status "Making scripts executable..."
chmod +x run_integrated_rag.py
chmod +x setup_rag_system.sh
if [[ -f "keystroke_logger.py" ]]; then
    chmod +x keystroke_logger.py
fi
print_success "Scripts made executable"

# Test basic functionality
print_status "Testing basic system components..."

# Test import of core modules
if python3 -c "
try:
    from ollama_rag_system import RAGSystem
    from embedding_pipeline import EmbeddingPipeline
    from rag_query_interface import RAGQueryInterface
    print('✅ Core modules imported successfully')
except ImportError as e:
    print('❌ Module import failed:', e)
    exit(1)
" 2>/dev/null; then
    print_success "Core modules test passed"
else
    print_error "Core modules test failed"
    print_error "Please check that all Python files are present and syntax is correct"
fi

# Check Ollama availability
print_status "Checking Ollama availability..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_success "✅ Ollama is running and accessible"
    
    # Check for embedding model
    if curl -s http://localhost:11434/api/tags | grep -q "nomic-embed-text"; then
        print_success "✅ nomic-embed-text model available"
    else
        print_warning "⚠️  nomic-embed-text model not found"
        print_warning "    Install with: ollama pull nomic-embed-text"
    fi
else
    print_warning "⚠️  Ollama not running or not accessible"
    print_warning "    Make sure Ollama is installed and running: ollama serve"
fi

# Create launcher scripts
print_status "Creating launcher scripts..."

# GUI launcher
cat > launch_rag_gui.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 run_integrated_rag.py --gui
EOF
chmod +x launch_rag_gui.sh

# CLI launcher  
cat > launch_rag_cli.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 run_integrated_rag.py --cli
EOF
chmod +x launch_rag_cli.sh

# Daemon launcher
cat > launch_rag_daemon.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 run_integrated_rag.py --daemon
EOF
chmod +x launch_rag_daemon.sh

print_success "Launcher scripts created"

# Create README
print_status "Creating documentation..."
cat > README_RAG_SYSTEM.md << 'EOF'
# Integrated RAG System

A comprehensive Retrieval-Augmented Generation system that combines real-time keystroke logging, OCR analysis, embedding generation, and intelligent querying for enhanced AI interactions.

## Features

- **Real-time Keystroke Logging**: Secure capture with privacy controls
- **OCR Integration**: Screen text analysis and processing  
- **Embedding Generation**: Multiple backend support (Ollama, Sentence Transformers, fallback)
- **Vector Database**: ChromaDB or SQLite storage for semantic search
- **Intelligent Querying**: Context-aware AI responses with historical data
- **Privacy Controls**: Comprehensive data filtering and anonymization
- **GUI & CLI Interfaces**: Multiple interaction modes

## Quick Start

1. **Setup**: Run `./setup_rag_system.sh`
2. **Launch GUI**: `./launch_rag_gui.sh`
3. **Launch CLI**: `./launch_rag_cli.sh`  
4. **Run Daemon**: `./launch_rag_daemon.sh`

## Manual Usage

```bash
# Initialize only
python3 run_integrated_rag.py --init-only

# Launch with GUI
python3 run_integrated_rag.py --gui

# Launch with CLI
python3 run_integrated_rag.py --cli

# Run as daemon
python3 run_integrated_rag.py --daemon
```

## Configuration

Edit `rag_config.json` to customize:
- Storage paths
- Keystroke logging settings
- Embedding pipeline parameters
- OCR integration options
- GUI/CLI preferences
- RAG system behavior

## Requirements

### Required
- Python 3.8+
- requests
- numpy

### Optional (Recommended)
- chromadb (vector storage)
- sentence-transformers (embeddings)
- faiss-cpu (similarity search)
- tiktoken (token counting)
- pynput (keystroke logging)
- PyQt5 (enhanced GUI)

### System Dependencies
- X11 development headers (Linux)
- Ollama server (for LLM integration)

## Privacy & Security

- Privacy mode enabled by default
- Sensitive data filtering
- Session-based anonymization
- Secure local storage only
- No network data transmission (except Ollama API)

## Troubleshooting

1. **Import Errors**: Ensure virtual environment is activated
2. **Keystroke Logging Issues**: Check X11 dependencies and permissions
3. **Ollama Connection**: Verify Ollama is running on localhost:11434
4. **GUI Issues**: Install PyQt5 or use CLI interface
5. **Performance**: Adjust chunk sizes and cache settings in config

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Keystroke Logger│    │ OCR Assistant    │    │ Query Interface │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          ▼                      ▼                       ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                Embedding Pipeline                           │
    └─────────────────────┬───────────────────────────────────────┘
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  RAG System                                 │
    │  ┌─────────────────┐    ┌──────────────────────────────┐    │
    │  │ Vector Database │    │ Similarity Search & Ranking │    │
    │  └─────────────────┘    └──────────────────────────────┘    │
    └─────────────────────┬───────────────────────────────────────┘
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Ollama LLM Integration                         │
    └─────────────────────────────────────────────────────────────┘
```
EOF

print_success "Documentation created: README_RAG_SYSTEM.md"

# Final status
echo ""
echo "🎉 RAG System Setup Complete!"
echo "=============================="
print_success "✅ Virtual environment created and activated"
print_success "✅ Dependencies installed (with fallbacks for optional packages)"
print_success "✅ Storage directories created"
print_success "✅ Configuration file created"
print_success "✅ Launcher scripts created"
print_success "✅ Documentation generated"

echo ""
echo "🚀 Next Steps:"
echo "  1. Start Ollama: ollama serve"
echo "  2. Pull embedding model: ollama pull nomic-embed-text"
echo "  3. Launch GUI: ./launch_rag_gui.sh"
echo "  4. Or CLI: ./launch_rag_cli.sh"
echo ""
echo "📖 Read README_RAG_SYSTEM.md for detailed usage instructions"
echo ""

# Show configuration preview
echo "⚙️  Current Configuration Preview:"
echo "   Storage: $(pwd)/rag_storage"
echo "   Keystroke Logging: Enabled (Privacy Mode)"
echo "   OCR Integration: Enabled"
echo "   GUI Interface: Enabled"
echo ""

# Check if user wants to launch immediately
echo "Would you like to launch the RAG system now? (GUI/cli/daemon/no)"
read -p "Choice [no]: " launch_choice

case "${launch_choice,,}" in
    gui|g)
        print_status "Launching GUI interface..."
        ./launch_rag_gui.sh
        ;;
    cli|c)
        print_status "Launching CLI interface..."
        ./launch_rag_cli.sh
        ;;
    daemon|d)
        print_status "Launching daemon mode..."
        ./launch_rag_daemon.sh
        ;;
    *)
        print_status "Setup complete. Use launcher scripts when ready."
        ;;
esac