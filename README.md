# Modular RAG System

A comprehensive, production-ready Retrieval-Augmented Generation (RAG) system with file upload capabilities, real-time processing, and web-accessible links for LLM integration.

## Features

### **File Upload System**
- **Folder-based uploads**: Drop JSON/TXT files into organized folders
- **Web-accessible links**: Direct HTTP links for LLM access
- **Real-time processing**: Files are automatically embedded and indexed
- **Multiple formats**: JSON (structured data) and TXT (documentation) support

### **Advanced RAG Pipeline**
- **Multi-backend embeddings**: Ollama → Sentence Transformers → Fallback
- **Smart text chunking**: Optimized for different content types
- **Vector storage**: ChromaDB with SQLite fallback
- **Similarity search**: FAISS-powered semantic matching

### **Privacy & Security**
- **Local-only processing**: No external data transmission
- **Privacy controls**: Sensitive data filtering and anonymization
- **Secure storage**: Encrypted local file system
- **Access controls**: Configurable permissions

### **Web Integration**
- **HTTP file server**: Built-in web server for file access
- **REST API**: JSON endpoints for file listing and metadata
- **CORS support**: Cross-origin resource sharing enabled
- **Direct LLM access**: Simple links for AI model consumption

## Quick Start

### 1. Setup
```bash
# Clone or download the package
cd Modular_RAG_Package

# Run the setup script
chmod +x setup_rag_system.sh
./setup_rag_system.sh
```

### 2. Launch the System
```bash
# Simple launcher
python3 rag_launcher.py

# Or launch directly
python3 enhanced_rag_system.py --port 8089
```

### 3. Upload Files
```bash
# Place your files in the upload folders
uploads/json/ # For JSON files
uploads/txt/ # For text files

# Files are automatically processed and available via web links
```

### 4. Query and Access
```python
# Query the system
response = system.query_with_files("What project information do you have?")

# Get direct links for LLM access
for ref in response['file_references']:
 print(f"File: {ref['filename']}")
 print(f"Link: {ref['access_link']}")
```

## ️ Installation

### Requirements
- **Python 3.8+**
- **Base packages**: `requests`, `numpy`, `pathlib`

### Optional (Recommended)
- **ChromaDB**: `pip install chromadb` (vector storage)
- **Sentence Transformers**: `pip install sentence-transformers` (embeddings)
- **FAISS**: `pip install faiss-cpu` (similarity search)
- **tiktoken**: `pip install tiktoken` (token counting)

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install libx11-dev libxext-dev

# For enhanced features
sudo apt-get install curl git python3-venv
```

## Usage Examples

### Basic File Upload and Query
```python
from enhanced_rag_system import EnhancedRAGSystem

# Initialize system
rag = EnhancedRAGSystem("./storage", port=8089)
rag.start()

# Add a file
uploaded_file = rag.add_file_from_path("document.json")
print(f"File available at: {rag.file_manager.get_file_link(uploaded_file.file_id)}")

# Query with file references
response = rag.query_with_files("Show me the documentation")
for ref in response['file_references']:
 print(f" {ref['filename']} -> {ref['access_link']}")
```

### Web Interface Usage
```bash
# Start system
python3 rag_launcher.py --system enhanced --port 8089

# Access web interface
curl http/localhost:8089/files # List all files
curl http/localhost:8089/files/[file_id]/[filename] # Access specific file
```

### Integration with LLM
```python
# The LLM can directly access files via returned links
query_response = rag.query_with_files("What's in the project documentation?")

# Extract links for LLM context
file_links = [ref['access_link'] for ref in query_response['file_references']]

# Pass to your LLM
llm_prompt = f"""
Based on the query: {query_response['query']}

Available documents:
{chr(10).join(f"- {ref['filename']}: {ref['access_link']}" for ref in query_response['file_references'])}

Please analyze the content and provide insights.
"""
```

## ️ Architecture

```
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ File Upload │ │ Web Server │ │ Query Interface │
│ Manager │ │ (Port 8089) │ │ │
└─────────┬───────┘ └─────────┬────────┘ └─────────┬───────┘
 │ │ │
 ▼ ▼ ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Embedding Pipeline │
 │ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
 │ │ Ollama │ │ Sentence │ │ Fallback │ │
 │ │ Embeddings │ │Transformers │ │ Embedder │ │
 │ └─────────────┘ └─────────────┘ └─────────────────────┘ │
 └─────────────────────┬───────────────────────────────────────┘
 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ RAG System Core │
 │ ┌─────────────────┐ ┌──────────────────────────────┐ │
 │ │ Vector Database │ │ Similarity Search & Ranking │ │
 │ │ (ChromaDB) │ │ (FAISS) │ │
 │ └─────────────────┘ └──────────────────────────────┘ │
 └─────────────────────────────────────────────────────────────┘
```

## Configuration

### Basic Configuration (`rag_config.json`)
```json
{
 "storage_path": "./rag_storage",
 "file_upload": {
 "port": 8089,
 "upload_folder": "uploads",
 "supported_formats": [".json", ".txt"],
 "max_file_size": "10MB"
 },
 "embedding_pipeline": {
 "chunk_size": 512,
 "cache_size": 1000,
 "batch_timeout": 30
 },
 "rag_system": {
 "vector_backend": "auto",
 "max_documents": 10000,
 "similarity_threshold": 0.7
 }
}
```

### Advanced Features
```python
# Custom embedding configuration
system = EnhancedRAGSystem(
 storage_path="./custom_storage",
 upload_port=9090
)

# Add files programmatically
file_info = system.add_file_from_path("/path/to/document.json")

# Query with custom parameters
response = system.query_with_files(
 query="Find project documentation",
 max_results=10
)
```

## API Reference

### EnhancedRAGSystem Class
```python
class EnhancedRAGSystem:
 def __init__(self, storage_path: Path, upload_port: int = 8089)
 def start() # Start all system components
 def stop() # Stop all system components
 def query_with_files(self, query: str, max_results: int = 5) -> Dict
 def add_file_from_path(self, file_path: str) -> UploadedFile
 def get_system_stats() -> Dict
```

### Web API Endpoints
```http
GET /files # List all uploaded files
GET /files/{file_id}/{filename} # Download specific file
```

### Response Format
```python
{
 'query': str, # Original query
 'rag_results': RAGResult, # Semantic search results
 'file_references': List[Dict], # Direct file links
 'processing_time': float, # Query processing time
 'total_documents': int, # Total embedded documents
 'file_links_available': bool # Whether file links exist
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
 ```bash
 # Solution: Activate virtual environment
 source venv/bin/activate
 ```

2. **Port Already in Use**
 ```python
 # Solution: Use different port
 system = EnhancedRAGSystem(port=9090)
 ```

3. **File Processing Errors**
 ```bash
 # Check file format and encoding
 file -bi your_file.json # Should show UTF-8
 ```

4. **Web Server Issues**
 ```bash
 # Check firewall and permissions
 sudo ufw allow 8089
 ```

### Performance Tuning
```python
# Adjust chunk size for better performance
embedding_pipeline = EmbeddingPipeline(chunk_size=256) # Smaller chunks

# Increase cache for faster queries
system = EnhancedRAGSystem(cache_size=2000)

# Optimize for specific use case
config = {
 "chunk_size": 512, # Standard documents
 "max_results": 5, # Focused results
 "similarity_threshold": 0.75 # Higher precision
}
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- **Documentation**: [Full API docs](docs/)
- **Examples**: [Usage examples](examples/)
- **Issues**: [Report bugs](issues/)
- **Discussions**: [Community discussions](discussions/)

---

**Built for superhuman productivity at skill level 10000**

*Combining real-time data ingestion, intelligent semantic search, and web-accessible file links for seamless LLM integration.*