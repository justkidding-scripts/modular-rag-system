#!/usr/bin/env python3
"""
Enhanced RAG System with File Upload and Web Access
Extends the base RAG system with file upload capabilities and web link access
"""

import json
import time
import os
import shutil
import hashlib
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import mimetypes
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
from collections import defaultdict, deque

# Import our base RAG components
from ollama_rag_system import RAGSystem, RAGDocument, RAGResult
from embedding_pipeline import EmbeddingPipeline


@dataclass
class UploadedFile:
    """Represents an uploaded file in the system"""
    file_id: str
    filename: str
    file_path: Path
    content_type: str
    size_bytes: int
    upload_time: float
    processed: bool = False
    embedding_count: int = 0
    metadata: Dict[str, Any] = None
    access_link: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.access_link:
            self.access_link = f"/files/{self.file_id}/{self.filename}"


class FileUploadManager:
    """Manages file uploads and provides web access"""
    
    def __init__(self, upload_dir: Path, port: int = 8089):
        self.upload_dir = Path(upload_dir)
        self.port = port
        self.uploaded_files = {}
        self.file_links = {}
        self.logger = logging.getLogger("FileUploadManager")
        
        # Create upload directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "json").mkdir(exist_ok=True)
        (self.upload_dir / "txt").mkdir(exist_ok=True)
        (self.upload_dir / "processed").mkdir(exist_ok=True)
        
        # HTTP server for file access
        self.http_server = None
        self.server_thread = None
        self.running = False
        
        print(f"📁 File Upload Manager initialized: {self.upload_dir}")
        print(f"🌐 Web access will be available on port {self.port}")
    
    def start_web_server(self):
        """Start the web server for file access"""
        try:
            # Custom handler that serves files from upload directory
            class FileHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, upload_dir=None, file_manager=None, **kwargs):
                    self.upload_dir = upload_dir
                    self.file_manager = file_manager
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    """Handle GET requests for files"""
                    if self.path.startswith('/files/'):
                        # Extract file ID from path
                        path_parts = self.path.strip('/').split('/')
                        if len(path_parts) >= 3:
                            file_id = path_parts[1]
                            filename = '/'.join(path_parts[2:])
                            
                            if file_id in self.file_manager.uploaded_files:
                                uploaded_file = self.file_manager.uploaded_files[file_id]
                                
                                try:
                                    # Serve the file
                                    with open(uploaded_file.file_path, 'rb') as f:
                                        content = f.read()
                                    
                                    self.send_response(200)
                                    self.send_header('Content-Type', uploaded_file.content_type)
                                    self.send_header('Content-Length', str(len(content)))
                                    self.send_header('Access-Control-Allow-Origin', '*')
                                    self.end_headers()
                                    self.wfile.write(content)
                                    return
                                except Exception as e:
                                    self.send_error(500, f"Error reading file: {e}")
                                    return
                    
                    elif self.path == '/' or self.path == '/files':
                        # List available files
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        file_list = []
                        for file_id, uploaded_file in self.file_manager.uploaded_files.items():
                            file_list.append({
                                'file_id': file_id,
                                'filename': uploaded_file.filename,
                                'access_link': f"http://localhost:{self.file_manager.port}{uploaded_file.access_link}",
                                'content_type': uploaded_file.content_type,
                                'size_bytes': uploaded_file.size_bytes,
                                'upload_time': uploaded_file.upload_time,
                                'processed': uploaded_file.processed
                            })
                        
                        response = json.dumps(file_list, indent=2)
                        self.wfile.write(response.encode())
                        return
                    
                    # Default 404
                    self.send_error(404, "File not found")
            
            # Create handler with our parameters
            handler = lambda *args, **kwargs: FileHandler(*args, upload_dir=self.upload_dir, file_manager=self, **kwargs)
            
            # Start server
            self.http_server = HTTPServer(('localhost', self.port), handler)
            self.running = True
            
            def serve_forever():
                self.logger.info(f"Web server started on http://localhost:{self.port}")
                self.http_server.serve_forever()
            
            self.server_thread = threading.Thread(target=serve_forever, daemon=True)
            self.server_thread.start()
            
            print(f"🌐 Web server started: http://localhost:{self.port}")
            print(f"📋 File list: http://localhost:{self.port}/files")
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            raise
    
    def stop_web_server(self):
        """Stop the web server"""
        if self.http_server:
            self.running = False
            self.http_server.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5)
            print("🛑 Web server stopped")
    
    def upload_folder_contents(self, folder_path: Path) -> List[UploadedFile]:
        """Upload all JSON and TXT files from a folder"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        uploaded_files = []
        supported_extensions = {'.json', '.txt'}
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    uploaded_file = self.add_file(file_path)
                    uploaded_files.append(uploaded_file)
                except Exception as e:
                    self.logger.error(f"Failed to upload {file_path}: {e}")
        
        print(f"📁 Uploaded {len(uploaded_files)} files from {folder_path}")
        return uploaded_files
    
    def add_file(self, file_path: Path, content_type: str = None) -> UploadedFile:
        """Add a file to the upload system"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())[:8]
        
        # Determine content type
        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                if file_path.suffix.lower() == '.json':
                    content_type = 'application/json'
                elif file_path.suffix.lower() == '.txt':
                    content_type = 'text/plain'
                else:
                    content_type = 'application/octet-stream'
        
        # Determine storage location based on type
        if content_type == 'application/json':
            storage_dir = self.upload_dir / "json"
        elif content_type.startswith('text/'):
            storage_dir = self.upload_dir / "txt"
        else:
            storage_dir = self.upload_dir / "processed"
        
        # Copy file to storage location
        stored_path = storage_dir / f"{file_id}_{file_path.name}"
        shutil.copy2(file_path, stored_path)
        
        # Create uploaded file record
        uploaded_file = UploadedFile(
            file_id=file_id,
            filename=file_path.name,
            file_path=stored_path,
            content_type=content_type,
            size_bytes=stored_path.stat().st_size,
            upload_time=time.time(),
            metadata={
                'original_path': str(file_path),
                'file_hash': self._calculate_file_hash(stored_path)
            }
        )
        
        # Store in registry
        self.uploaded_files[file_id] = uploaded_file
        self.file_links[str(file_path)] = uploaded_file.access_link
        
        self.logger.info(f"File added: {file_path.name} -> {uploaded_file.access_link}")
        return uploaded_file
    
    def get_file_link(self, file_id: str) -> Optional[str]:
        """Get web link for a file"""
        if file_id in self.uploaded_files:
            uploaded_file = self.uploaded_files[file_id]
            return f"http://localhost:{self.port}{uploaded_file.access_link}"
        return None
    
    def list_files(self) -> List[UploadedFile]:
        """List all uploaded files"""
        return list(self.uploaded_files.values())
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class EnhancedRAGSystem:
    """Enhanced RAG system with file upload capabilities"""
    
    def __init__(self, storage_path: Path, upload_port: int = 8089):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize upload folder
        self.upload_folder = self.storage_path / "uploads"
        self.upload_folder.mkdir(exist_ok=True)
        
        # Create example folders for users
        (self.upload_folder / "json").mkdir(exist_ok=True)
        (self.upload_folder / "txt").mkdir(exist_ok=True)
        
        # Initialize components
        self.base_rag = RAGSystem(self.storage_path / "rag_data")
        self.embedding_pipeline = EmbeddingPipeline(self.storage_path)
        self.file_manager = FileUploadManager(self.upload_folder, upload_port)
        
        self.logger = logging.getLogger("EnhancedRAGSystem")
        
        # Create example files
        self._create_example_files()
        
        print(f"🚀 Enhanced RAG System initialized")
        print(f"📁 Upload folder: {self.upload_folder}")
        print(f"🌐 Web access: http://localhost:{upload_port}")
    
    def start(self):
        """Start all system components"""
        self.file_manager.start_web_server()
        
        # Process any existing files in upload folder
        uploaded_files = self.file_manager.upload_folder_contents(self.upload_folder)
        
        # Process uploaded files into RAG system
        for uploaded_file in uploaded_files:
            self.process_uploaded_file(uploaded_file)
        
        print("✅ Enhanced RAG System started")
        print(f"📋 File listing: http://localhost:{self.file_manager.port}/files")
        print(f"📁 Place your JSON/TXT files in: {self.upload_folder}")
    
    def stop(self):
        """Stop all system components"""
        self.file_manager.stop_web_server()
        self.base_rag.shutdown()
        print("🛑 Enhanced RAG System stopped")
    
    def process_uploaded_file(self, uploaded_file: UploadedFile):
        """Process an uploaded file into the RAG system"""
        try:
            # Read file content
            with open(uploaded_file.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                **uploaded_file.metadata,
                'file_id': uploaded_file.file_id,
                'filename': uploaded_file.filename,
                'content_type': uploaded_file.content_type,
                'access_link': uploaded_file.access_link,
                'upload_time': uploaded_file.upload_time
            }
            
            # Generate embeddings
            embedding_results = self.embedding_pipeline.process_content(
                content, 'document', metadata
            )
            
            # Convert to RAG documents
            rag_documents = self.embedding_pipeline.create_rag_documents(embedding_results)
            
            # Add to RAG system
            self.base_rag.add_documents(rag_documents)
            
            # Update file record
            uploaded_file.processed = True
            uploaded_file.embedding_count = len(embedding_results)
            
            self.logger.info(f"Processed file: {uploaded_file.filename} -> {len(embedding_results)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to process file {uploaded_file.filename}: {e}")
    
    def query_with_files(self, query: str, max_results: int = 5) -> Dict:
        """Enhanced query that includes file references"""
        start_time = time.time()
        
        # Basic RAG query
        rag_result = self.base_rag.query(query, max_results=max_results)
        
        # Find relevant uploaded files
        file_references = self._find_file_references(query)
        
        # Enhanced response
        response = {
            'query': query,
            'rag_results': rag_result,
            'file_references': file_references,
            'processing_time': time.time() - start_time,
            'total_documents': len(rag_result.documents),
            'file_links_available': len(file_references) > 0
        }
        
        return response
    
    def _find_file_references(self, query: str) -> List[Dict]:
        """Find references to uploaded files relevant to query"""
        query_words = set(query.lower().split())
        references = []
        
        for uploaded_file in self.file_manager.list_files():
            # Check filename relevance
            filename_words = set(uploaded_file.filename.lower().replace('.', ' ').split())
            filename_overlap = len(query_words.intersection(filename_words))
            
            if filename_overlap > 0:
                references.append({
                    'file_id': uploaded_file.file_id,
                    'filename': uploaded_file.filename,
                    'access_link': f"http://localhost:{self.file_manager.port}{uploaded_file.access_link}",
                    'relevance_score': filename_overlap / len(query_words),
                    'content_type': uploaded_file.content_type,
                    'processed': uploaded_file.processed,
                    'embedding_count': uploaded_file.embedding_count
                })
        
        return sorted(references, key=lambda x: x['relevance_score'], reverse=True)[:3]
    
    def add_file_from_path(self, file_path: str) -> UploadedFile:
        """Add a file from external path"""
        uploaded_file = self.file_manager.add_file(Path(file_path))
        self.process_uploaded_file(uploaded_file)
        return uploaded_file
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        rag_stats = self.base_rag.get_system_stats()
        embedding_stats = self.embedding_pipeline.get_stats()
        file_stats = {
            'total_files': len(self.file_manager.uploaded_files),
            'processed_files': sum(1 for f in self.file_manager.list_files() if f.processed),
            'total_embeddings': sum(f.embedding_count for f in self.file_manager.list_files()),
            'web_server_running': self.file_manager.running,
            'upload_folder': str(self.upload_folder)
        }
        
        return {
            'rag_system': rag_stats,
            'embedding_pipeline': embedding_stats,
            'file_system': file_stats,
            'web_access_port': self.file_manager.port
        }
    
    def _create_example_files(self):
        """Create example files for users"""
        # Example JSON file
        example_json = {
            "project": "Enhanced RAG System",
            "description": "Advanced RAG system with file upload capabilities",
            "features": [
                "File upload system with web links",
                "JSON and TXT file processing", 
                "Automatic embedding generation",
                "Web-accessible file links",
                "Real-time file processing"
            ],
            "usage": {
                "step1": "Place JSON files in uploads/json/ folder",
                "step2": "Place TXT files in uploads/txt/ folder", 
                "step3": "Files are automatically processed and embedded",
                "step4": "Query the system to get relevant file links",
                "step5": "Access files via http://localhost:8089/files/[file_id]/[filename]"
            },
            "tips": [
                "Supported formats: JSON, TXT",
                "Files are processed automatically on startup",
                "Use descriptive filenames for better relevance matching",
                "Check http://localhost:8089/files for file listing"
            ]
        }
        
        json_file = self.upload_folder / "json" / "rag_system_info.json"
        if not json_file.exists():
            with open(json_file, 'w') as f:
                json.dump(example_json, f, indent=2)
        
        # Example TXT file
        txt_content = """Enhanced RAG System - Usage Guide

OVERVIEW:
This enhanced RAG system provides file upload capabilities with web-accessible links.
The LLM can now reference and provide direct links to your uploaded documents.

FILE UPLOAD PROCESS:
1. Place JSON files in: uploads/json/
2. Place TXT files in: uploads/txt/
3. System automatically processes files on startup
4. Files become searchable via embeddings
5. Direct web links are generated for each file

WEB ACCESS:
- File listing: http://localhost:8089/files
- Individual files: http://localhost:8089/files/[file_id]/[filename]
- CORS enabled for external access
- JSON API response with file metadata

QUERYING:
When you query the system, it will:
- Search through embedded file content
- Return relevant file references with direct links
- Provide relevance scores for file matches
- Include file metadata (type, size, processing status)

EXAMPLES:
Query: "What project information do you have?"
Response: May include link to rag_system_info.json

Query: "Show me documentation about file uploads"
Response: May include link to this usage guide

SUPPORTED FORMATS:
- JSON files (.json) - Structured data, configuration files
- Text files (.txt) - Documentation, notes, guides
- Automatic content type detection
- UTF-8 encoding support

TIPS FOR BEST RESULTS:
- Use descriptive filenames
- Include relevant keywords in content
- Organize files by type (json vs txt folders)
- Check processing status via web interface
- Files are permanently stored until manually removed
"""
        
        txt_file = self.upload_folder / "txt" / "usage_guide.txt"
        if not txt_file.exists():
            with open(txt_file, 'w') as f:
                f.write(txt_content)


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAG System with File Upload")
    parser.add_argument("--storage", type=Path, default="./enhanced_rag_storage", help="Storage path")
    parser.add_argument("--port", type=int, default=8089, help="Web server port")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    # Create enhanced RAG system
    enhanced_rag = EnhancedRAGSystem(args.storage, args.port)
    
    try:
        enhanced_rag.start()
        
        if args.test:
            print("\n🧪 Running Enhanced RAG Tests...")
            
            # Test file listing
            print("\n📁 Testing file system...")
            files = enhanced_rag.file_manager.list_files()
            for file_info in files:
                print(f"   📄 {file_info.filename} -> {enhanced_rag.file_manager.get_file_link(file_info.file_id)}")
            
            # Test enhanced query
            print("\n🔍 Testing enhanced query with file references...")
            response = enhanced_rag.query_with_files("How do I use the file upload system?")
            
            print(f"   RAG Results: {len(response['rag_results'].documents)}")
            print(f"   File References: {len(response['file_references'])}")
            print(f"   Processing Time: {response['processing_time']:.3f}s")
            
            if response['file_references']:
                print("   📎 File Links:")
                for ref in response['file_references']:
                    print(f"     • {ref['filename']} ({ref['relevance_score']:.2f}) -> {ref['access_link']}")
            
            # Test system stats
            print("\n📊 System Statistics:")
            stats = enhanced_rag.get_system_stats()
            print(f"   Files: {stats['file_system']['total_files']} ({stats['file_system']['processed_files']} processed)")
            print(f"   Embeddings: {stats['file_system']['total_embeddings']}")
            print(f"   Web Server: {'Running' if stats['file_system']['web_server_running'] else 'Stopped'}")
            
            print("\n✅ All tests completed!")
        
        else:
            print(f"\n🚀 Enhanced RAG System running on http://localhost:{args.port}")
            print("📁 Upload your JSON/TXT files to the uploads/ folder")
            print("🔍 Query the system to get file links and content")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    finally:
        enhanced_rag.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced RAG System with File Upload and Web Access
Extends the base RAG system with file upload capabilities and web link access
"""

import json
import time
import os
import shutil
import hashlib
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import mimetypes
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
from collections import defaultdict, deque

# Import our base RAG components
from ollama_rag_system import RAGSystem, RAGDocument, RAGResult
from embedding_pipeline import EmbeddingPipeline


@dataclass
class UploadedFile:
    """Represents an uploaded file in the system"""
    file_id: str
    filename: str
    file_path: Path
    content_type: str
    size_bytes: int
    upload_time: float
    processed: bool = False
    embedding_count: int = 0
    metadata: Dict[str, Any] = None
    access_link: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.access_link:
            self.access_link = f"/files/{self.file_id}/{self.filename}"


@dataclass
class CrossAppContext:
    """Cross-application context bridge data"""
    app_name: str
    window_title: str
    activity_type: str
    content_snippet: str
    timestamp: float
    related_files: List[str] = None
    context_bridge_score: float = 0.0
    
    def __post_init__(self):
        if self.related_files is None:
            self.related_files = []


class FileUploadManager:
    """Manages file uploads and provides web access"""
    
    def __init__(self, upload_dir: Path, port: int = 8089):
        self.upload_dir = Path(upload_dir)
        self.port = port
        self.uploaded_files = {}
        self.file_links = {}
        self.logger = logging.getLogger("FileUploadManager")
        
        # Create upload directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "json").mkdir(exist_ok=True)
        (self.upload_dir / "txt").mkdir(exist_ok=True)
        (self.upload_dir / "processed").mkdir(exist_ok=True)
        
        # HTTP server for file access
        self.http_server = None
        self.server_thread = None
        self.running = False
        
        print(f"📁 File Upload Manager initialized: {self.upload_dir}")
        print(f"🌐 Web access will be available on port {self.port}")
    
    def start_web_server(self):
        """Start the web server for file access"""
        try:
            # Custom handler that serves files from upload directory
            class FileHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, upload_dir=None, file_manager=None, **kwargs):
                    self.upload_dir = upload_dir
                    self.file_manager = file_manager
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    """Handle GET requests for files"""
                    if self.path.startswith('/files/'):
                        # Extract file ID from path
                        path_parts = self.path.strip('/').split('/')
                        if len(path_parts) >= 3:
                            file_id = path_parts[1]
                            filename = '/'.join(path_parts[2:])
                            
                            if file_id in self.file_manager.uploaded_files:
                                uploaded_file = self.file_manager.uploaded_files[file_id]
                                
                                try:
                                    # Serve the file
                                    with open(uploaded_file.file_path, 'rb') as f:
                                        content = f.read()
                                    
                                    self.send_response(200)
                                    self.send_header('Content-Type', uploaded_file.content_type)
                                    self.send_header('Content-Length', str(len(content)))
                                    self.send_header('Access-Control-Allow-Origin', '*')
                                    self.end_headers()
                                    self.wfile.write(content)
                                    return
                                except Exception as e:
                                    self.send_error(500, f"Error reading file: {e}")
                                    return
                    
                    elif self.path == '/' or self.path == '/files':
                        # List available files
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        file_list = []
                        for file_id, uploaded_file in self.file_manager.uploaded_files.items():
                            file_list.append({
                                'file_id': file_id,
                                'filename': uploaded_file.filename,
                                'access_link': f"http://localhost:{self.file_manager.port}{uploaded_file.access_link}",
                                'content_type': uploaded_file.content_type,
                                'size_bytes': uploaded_file.size_bytes,
                                'upload_time': uploaded_file.upload_time,
                                'processed': uploaded_file.processed
                            })
                        
                        response = json.dumps(file_list, indent=2)
                        self.wfile.write(response.encode())
                        return
                    
                    # Default 404
                    self.send_error(404, "File not found")
            
            # Create handler with our parameters
            handler = lambda *args, **kwargs: FileHandler(*args, upload_dir=self.upload_dir, file_manager=self, **kwargs)
            
            # Start server
            self.http_server = HTTPServer(('localhost', self.port), handler)
            self.running = True
            
            def serve_forever():
                self.logger.info(f"Web server started on http://localhost:{self.port}")
                self.http_server.serve_forever()
            
            self.server_thread = threading.Thread(target=serve_forever, daemon=True)
            self.server_thread.start()
            
            print(f"🌐 Web server started: http://localhost:{self.port}")
            print(f"📋 File list: http://localhost:{self.port}/files")
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            raise
    
    def stop_web_server(self):
        """Stop the web server"""
        if self.http_server:
            self.running = False
            self.http_server.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5)
            print("🛑 Web server stopped")
    
    def add_file(self, file_path: Path, content_type: str = None) -> UploadedFile:
        """Add a file to the upload system"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())[:8]
        
        # Determine content type
        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                if file_path.suffix.lower() == '.json':
                    content_type = 'application/json'
                elif file_path.suffix.lower() == '.txt':
                    content_type = 'text/plain'
                else:
                    content_type = 'application/octet-stream'
        
        # Determine storage location based on type
        if content_type == 'application/json':
            storage_dir = self.upload_dir / "json"
        elif content_type.startswith('text/'):
            storage_dir = self.upload_dir / "txt"
        else:
            storage_dir = self.upload_dir / "processed"
        
        # Copy file to storage location
        stored_path = storage_dir / f"{file_id}_{file_path.name}"
        shutil.copy2(file_path, stored_path)
        
        # Create uploaded file record
        uploaded_file = UploadedFile(
            file_id=file_id,
            filename=file_path.name,
            file_path=stored_path,
            content_type=content_type,
            size_bytes=stored_path.stat().st_size,
            upload_time=time.time(),
            metadata={
                'original_path': str(file_path),
                'file_hash': self._calculate_file_hash(stored_path)
            }
        )
        
        # Store in registry
        self.uploaded_files[file_id] = uploaded_file
        self.file_links[str(file_path)] = uploaded_file.access_link
        
        self.logger.info(f"File added: {file_path.name} -> {uploaded_file.access_link}")
        return uploaded_file
    
    def upload_folder_contents(self, folder_path: Path) -> List[UploadedFile]:
        """Upload all JSON and TXT files from a folder"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        uploaded_files = []
        supported_extensions = {'.json', '.txt'}
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    uploaded_file = self.add_file(file_path)
                    uploaded_files.append(uploaded_file)
                except Exception as e:
                    self.logger.error(f"Failed to upload {file_path}: {e}")
        
        print(f"📁 Uploaded {len(uploaded_files)} files from {folder_path}")
        return uploaded_files
    
    def get_file_link(self, file_id: str) -> Optional[str]:
        """Get web link for a file"""
        if file_id in self.uploaded_files:
            uploaded_file = self.uploaded_files[file_id]
            return f"http://localhost:{self.port}{uploaded_file.access_link}"
        return None
    
    def list_files(self) -> List[UploadedFile]:
        """List all uploaded files"""
        return list(self.uploaded_files.values())
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class CrossAppContextBridge:
    """Enhancement #8: Cross-Application Context Bridging"""
    
    def __init__(self, max_contexts: int = 1000):
        self.contexts = deque(maxlen=max_contexts)
        self.app_patterns = defaultdict(list)
        self.context_bridges = defaultdict(list)
        self.logger = logging.getLogger("CrossAppContextBridge")
        
        # Application context patterns
        self.app_context_patterns = {
            'vscode': ['coding', 'debugging', 'git'],
            'browser': ['research', 'documentation', 'stackoverflow'],
            'terminal': ['commands', 'scripts', 'system'],
            'slack': ['communication', 'collaboration', 'meetings'],
            'notion': ['notes', 'planning', 'documentation']
        }
    
    def add_context(self, app_name: str, window_title: str, activity_type: str, content_snippet: str):
        """Add new application context"""
        context = CrossAppContext(
            app_name=app_name.lower(),
            window_title=window_title,
            activity_type=activity_type,
            content_snippet=content_snippet[:500],  # Limit snippet size
            timestamp=time.time()
        )
        
        # Calculate context bridge score
        context.context_bridge_score = self._calculate_bridge_score(context)
        
        self.contexts.append(context)
        self.app_patterns[app_name.lower()].append(context)
        
        # Find related contexts
        self._find_related_contexts(context)
    
    def _calculate_bridge_score(self, context: CrossAppContext) -> float:
        """Calculate how likely this context is to bridge with others"""
        score = 0.0
        
        # App transition bonus
        recent_apps = [c.app_name for c in list(self.contexts)[-5:]]
        if context.app_name not in recent_apps:
            score += 0.3
        
        # Content complexity bonus
        content_words = len(context.content_snippet.split())
        score += min(content_words / 100, 0.4)
        
        # Pattern recognition bonus
        patterns = self.app_context_patterns.get(context.app_name, [])
        for pattern in patterns:
            if pattern in context.content_snippet.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    def _find_related_contexts(self, new_context: CrossAppContext):
        """Find contexts related to the new one"""
        for existing_context in list(self.contexts)[-20:]:  # Check last 20 contexts
            if existing_context.app_name != new_context.app_name:
                # Check for keyword overlap
                new_words = set(new_context.content_snippet.lower().split())
                existing_words = set(existing_context.content_snippet.lower().split())
                
                overlap = len(new_words.intersection(existing_words))
                if overlap >= 3:  # Minimum 3 words in common
                    bridge_key = f"{existing_context.app_name}-{new_context.app_name}"
                    self.context_bridges[bridge_key].append({
                        'context1': existing_context,
                        'context2': new_context,
                        'overlap_score': overlap / len(new_words.union(existing_words)),
                        'created_at': time.time()
                    })
    
    def get_bridged_context(self, current_app: str, query: str) -> List[Dict]:
        """Get context from other applications relevant to current query"""
        query_words = set(query.lower().split())
        bridges = []
        
        for bridge_key, bridge_list in self.context_bridges.items():
            if current_app.lower() in bridge_key:
                for bridge in bridge_list[-5:]:  # Recent bridges
                    context1, context2 = bridge['context1'], bridge['context2']
                    
                    # Check relevance to query
                    for context in [context1, context2]:
                        if context.app_name != current_app.lower():
                            context_words = set(context.content_snippet.lower().split())
                            relevance = len(query_words.intersection(context_words))
                            
                            if relevance >= 2:
                                bridges.append({
                                    'app_name': context.app_name,
                                    'content': context.content_snippet,
                                    'relevance_score': relevance / len(query_words),
                                    'bridge_score': bridge['overlap_score'],
                                    'timestamp': context.timestamp
                                })
        
        return sorted(bridges, key=lambda x: x['relevance_score'] + x['bridge_score'], reverse=True)[:5]


class PredictiveContextSwitcher:
    """Enhancement #2: Predictive Context Switching"""
    
    def __init__(self):
        self.switch_history = deque(maxlen=1000)
        self.patterns = defaultdict(list)
        self.predictions = {}
        self.logger = logging.getLogger("PredictiveContextSwitcher")
    
    def record_context_switch(self, from_app: str, to_app: str, trigger_content: str = ""):
        """Record a context switch event"""
        switch_event = {
            'from_app': from_app.lower(),
            'to_app': to_app.lower(),
            'trigger_content': trigger_content[:200],
            'timestamp': time.time(),
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
        
        self.switch_history.append(switch_event)
        
        # Update patterns
        pattern_key = f"{from_app}->{to_app}"
        self.patterns[pattern_key].append(switch_event)
        
        # Update predictions
        self._update_predictions()
    
    def _update_predictions(self):
        """Update prediction models based on patterns"""
        # Time-based patterns
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Recent pattern analysis (last 50 switches)
        recent_switches = list(self.switch_history)[-50:]
        
        for switch in recent_switches:
            # Time similarity
            if abs(switch['hour'] - current_hour) <= 1 and switch['day_of_week'] == current_day:
                pattern_key = f"{switch['from_app']}->{switch['to_app']}"
                
                if pattern_key not in self.predictions:
                    self.predictions[pattern_key] = {'count': 0, 'confidence': 0.0}
                
                self.predictions[pattern_key]['count'] += 1
                self.predictions[pattern_key]['confidence'] = min(
                    self.predictions[pattern_key]['count'] / 10, 0.9
                )
    
    def predict_next_context(self, current_app: str, current_content: str = "") -> List[Dict]:
        """Predict what context/app the user will switch to next"""
        predictions = []
        
        # Pattern-based predictions
        for pattern_key, data in self.predictions.items():
            if pattern_key.startswith(current_app.lower() + '->'):
                target_app = pattern_key.split('->')[-1]
                predictions.append({
                    'target_app': target_app,
                    'confidence': data['confidence'],
                    'reason': 'historical_pattern',
                    'pattern_strength': data['count']
                })
        
        # Content-based predictions
        content_words = set(current_content.lower().split())
        for switch in list(self.switch_history)[-20:]:
            if switch['from_app'] == current_app.lower():
                trigger_words = set(switch['trigger_content'].lower().split())
                overlap = len(content_words.intersection(trigger_words))
                
                if overlap >= 2:
                    predictions.append({
                        'target_app': switch['to_app'],
                        'confidence': min(overlap / 5, 0.8),
                        'reason': 'content_similarity',
                        'trigger_overlap': overlap
                    })
        
        # Sort by confidence and return top 3
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:3]


class LearningAdaptationSystem:
    """Enhancement #7: Real-time Learning Adaptation"""
    
    def __init__(self):
        self.feedback_history = deque(maxlen=500)
        self.response_improvements = defaultdict(list)
        self.learning_weights = {
            'positive_feedback': 1.2,
            'negative_feedback': 0.8,
            'ignored_response': 0.9,
            'followed_suggestion': 1.3
        }
        self.logger = logging.getLogger("LearningAdaptationSystem")
    
    def record_feedback(self, query: str, response: str, feedback_type: str, user_action: str = ""):
        """Record user feedback on AI response"""
        feedback = {
            'query': query,
            'response': response[:500],
            'feedback_type': feedback_type,
            'user_action': user_action,
            'timestamp': time.time(),
            'response_hash': hashlib.md5(response.encode()).hexdigest()[:8]
        }
        
        self.feedback_history.append(feedback)
        
        # Update learning weights for similar queries
        self._update_learning_weights(feedback)
    
    def _update_learning_weights(self, feedback: Dict):
        """Update learning weights based on feedback"""
        query_keywords = set(feedback['query'].lower().split())
        
        # Find similar past responses
        for past_feedback in list(self.feedback_history)[-50:]:
            past_keywords = set(past_feedback['query'].lower().split())
            similarity = len(query_keywords.intersection(past_keywords)) / len(query_keywords.union(past_keywords))
            
            if similarity > 0.3:  # Similar queries
                improvement_key = f"similarity_{similarity:.1f}"
                self.response_improvements[improvement_key].append({
                    'feedback_type': feedback['feedback_type'],
                    'improvement_factor': self.learning_weights.get(feedback['feedback_type'], 1.0),
                    'timestamp': feedback['timestamp']
                })
    
    def get_response_adjustments(self, query: str) -> Dict[str, float]:
        """Get response adjustments based on learning"""
        query_keywords = set(query.lower().split())
        adjustments = {
            'confidence_modifier': 1.0,
            'verbosity_modifier': 1.0,
            'suggestion_weight': 1.0
        }
        
        # Find relevant learning patterns
        for improvement_key, improvements in self.response_improvements.items():
            if improvements:
                recent_improvements = [i for i in improvements if time.time() - i['timestamp'] < 86400]  # Last 24h
                
                if recent_improvements:
                    avg_factor = sum(i['improvement_factor'] for i in recent_improvements) / len(recent_improvements)
                    
                    if 'similarity' in improvement_key:
                        adjustments['confidence_modifier'] *= avg_factor
                        
                        # Adjust based on feedback patterns
                        positive_count = sum(1 for i in recent_improvements if i['feedback_type'] == 'positive_feedback')
                        negative_count = sum(1 for i in recent_improvements if i['feedback_type'] == 'negative_feedback')
                        
                        if positive_count > negative_count:
                            adjustments['suggestion_weight'] *= 1.1
                        elif negative_count > positive_count:
                            adjustments['verbosity_modifier'] *= 0.8
        
        return adjustments


class EnhancedRAGSystem:
    """Main enhanced RAG system combining all enhancements"""
    
    def __init__(self, storage_path: Path, upload_port: int = 8089):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize upload folder
        self.upload_folder = self.storage_path / "uploads"
        self.upload_folder.mkdir(exist_ok=True)
        
        # Create example folders for users
        (self.upload_folder / "json").mkdir(exist_ok=True)
        (self.upload_folder / "txt").mkdir(exist_ok=True)
        
        # Initialize components
        self.base_rag = RAGSystem(self.storage_path / "rag_data")
        self.embedding_pipeline = EmbeddingPipeline(self.storage_path)
        self.file_manager = FileUploadManager(self.upload_folder, upload_port)
        self.context_bridge = CrossAppContextBridge()
        self.predictor = PredictiveContextSwitcher()
        self.learning_system = LearningAdaptationSystem()
        
        self.logger = logging.getLogger("EnhancedRAGSystem")
        
        # Create example files
        self._create_example_files()
        
        print(f"🚀 Enhanced RAG System initialized")
        print(f"📁 Upload folder: {self.upload_folder}")
        print(f"🌐 Web access: http://localhost:{upload_port}")
    
    def start(self):
        """Start all system components"""
        self.file_manager.start_web_server()
        
        # Process any existing files in upload folder
        uploaded_files = self.file_manager.upload_folder_contents(self.upload_folder)
        
        # Process uploaded files into RAG system
        for uploaded_file in uploaded_files:
            self.process_uploaded_file(uploaded_file)
        
        print("✅ Enhanced RAG System started")
    
    def stop(self):
        """Stop all system components"""
        self.file_manager.stop_web_server()
        self.base_rag.shutdown()
        print("🛑 Enhanced RAG System stopped")
    
    def process_uploaded_file(self, uploaded_file: UploadedFile):
        """Process an uploaded file into the RAG system"""
        try:
            # Read file content
            with open(uploaded_file.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                **uploaded_file.metadata,
                'file_id': uploaded_file.file_id,
                'filename': uploaded_file.filename,
                'content_type': uploaded_file.content_type,
                'access_link': uploaded_file.access_link,
                'upload_time': uploaded_file.upload_time
            }
            
            # Generate embeddings
            embedding_results = self.embedding_pipeline.process_content(
                content, 'document', metadata
            )
            
            # Convert to RAG documents
            rag_documents = self.embedding_pipeline.create_rag_documents(embedding_results)
            
            # Add to RAG system
            self.base_rag.add_documents(rag_documents)
            
            # Update file record
            uploaded_file.processed = True
            uploaded_file.embedding_count = len(embedding_results)
            
            self.logger.info(f"Processed file: {uploaded_file.filename} -> {len(embedding_results)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to process file {uploaded_file.filename}: {e}")
    
    def enhanced_query(self, query: str, current_app: str = "", current_content: str = "") -> Dict:
        """Enhanced query with all improvements"""
        start_time = time.time()
        
        # Get learning adjustments
        adjustments = self.learning_system.get_response_adjustments(query)
        
        # Basic RAG query
        rag_result = self.base_rag.query(query, max_results=5)
        
        # Cross-app context bridging
        bridged_contexts = self.context_bridge.get_bridged_context(current_app, query)
        
        # Predictive context switching
        next_context_predictions = self.predictor.predict_next_context(current_app, current_content)
        
        # Enhanced response
        response = {
            'query': query,
            'rag_results': rag_result,
            'bridged_contexts': bridged_contexts,
            'next_context_predictions': next_context_predictions,
            'learning_adjustments': adjustments,
            'processing_time': time.time() - start_time,
            'uploaded_file_references': self._find_file_references(query),
            'enhancement_insights': {
                'cross_app_insights': len(bridged_contexts),
                'predictive_confidence': max([p.get('confidence', 0) for p in next_context_predictions], default=0),
                'learning_factor': adjustments.get('confidence_modifier', 1.0)
            }
        }
        
        return response
    
    def _find_file_references(self, query: str) -> List[Dict]:
        """Find references to uploaded files relevant to query"""
        query_words = set(query.lower().split())
        references = []
        
        for uploaded_file in self.file_manager.list_files():
            # Check filename relevance
            filename_words = set(uploaded_file.filename.lower().replace('.', ' ').split())
            filename_overlap = len(query_words.intersection(filename_words))
            
            if filename_overlap > 0:
                references.append({
                    'file_id': uploaded_file.file_id,
                    'filename': uploaded_file.filename,
                    'access_link': f"http://localhost:{self.file_manager.port}{uploaded_file.access_link}",
                    'relevance_score': filename_overlap / len(query_words),
                    'content_type': uploaded_file.content_type,
                    'processed': uploaded_file.processed
                })
        
        return sorted(references, key=lambda x: x['relevance_score'], reverse=True)[:3]
    
    def add_context_switch(self, from_app: str, to_app: str, trigger_content: str = ""):
        """Record context switch for predictive learning"""
        self.predictor.record_context_switch(from_app, to_app, trigger_content)
        self.context_bridge.add_context(to_app, "", "context_switch", trigger_content)
    
    def record_feedback(self, query: str, response: str, feedback_type: str, user_action: str = ""):
        """Record user feedback for learning"""
        self.learning_system.record_feedback(query, response, feedback_type, user_action)
    
    def _create_example_files(self):
        """Create example files for users"""
        # Example JSON file
        example_json = {
            "project": "RAG System Enhancement",
            "features": [
                "File upload system",
                "Cross-application context bridging",
                "Predictive context switching",
                "Real-time learning adaptation"
            ],
            "instructions": "Place your JSON and TXT files in the uploads/json and uploads/txt folders. They will be automatically processed and made available via web links."
        }
        
        json_file = self.upload_folder / "json" / "example_project_info.json"
        if not json_file.exists():
            with open(json_file, 'w') as f:
                json.dump(example_json, f, indent=2)
        
        # Example TXT file
        txt_content = """Enhanced RAG System Usage Guide

This system provides advanced capabilities including:

1. File Upload System:
   - Place JSON files in uploads/json/
   - Place TXT files in uploads/txt/
   - Files are automatically processed and embedded
   - Access files via http://localhost:8089/files/[file_id]/[filename]

2. Cross-Application Context Bridging:
   - Tracks activity across different applications
   - Finds connections between work in different tools
   - Provides context-aware suggestions

3. Predictive Context Switching:
   - Learns your workflow patterns
   - Predicts what application you'll switch to next
   - Suggests optimal timing for context changes

4. Real-time Learning Adaptation:
   - Improves responses based on your feedback
   - Adapts to your preferences over time
   - Optimizes response style and confidence

To use these features, interact with the system through queries and provide feedback on the responses.
"""
        
        txt_file = self.upload_folder / "txt" / "usage_guide.txt"
        if not txt_file.exists():
            with open(txt_file, 'w') as f:
                f.write(txt_content)


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAG System")
    parser.add_argument("--storage", type=Path, default="./enhanced_rag_storage", help="Storage path")
    parser.add_argument("--port", type=int, default=8089, help="Web server port")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    # Create enhanced RAG system
    enhanced_rag = EnhancedRAGSystem(args.storage, args.port)
    
    try:
        enhanced_rag.start()
        
        if args.test:
            print("\n🧪 Running Enhanced RAG Tests...")
            
            # Test file upload
            print("\n📁 Testing file upload system...")
            files = enhanced_rag.file_manager.list_files()
            for file_info in files:
                print(f"   📄 {file_info.filename} -> {enhanced_rag.file_manager.get_file_link(file_info.file_id)}")
            
            # Test enhanced query
            print("\n🔍 Testing enhanced query...")
            response = enhanced_rag.enhanced_query(
                "How do I use the file upload system?",
                current_app="terminal",
                current_content="testing enhanced rag system"
            )
            
            print(f"   RAG Results: {len(response['rag_results'].documents)}")
            print(f"   Bridged Contexts: {response['enhancement_insights']['cross_app_insights']}")
            print(f"   File References: {len(response['uploaded_file_references'])}")
            print(f"   Processing Time: {response['processing_time']:.3f}s")
            
            # Test context switching
            print("\n🔄 Testing context switching...")
            enhanced_rag.add_context_switch("terminal", "browser", "looking up documentation")
            predictions = enhanced_rag.predictor.predict_next_context("terminal", "debugging code")
            print(f"   Predictions: {len(predictions)}")
            
            # Test learning system
            print("\n📚 Testing learning adaptation...")
            enhanced_rag.record_feedback(
                "test query", 
                "test response", 
                "positive_feedback", 
                "followed_suggestion"
            )
            adjustments = enhanced_rag.learning_system.get_response_adjustments("similar test query")
            print(f"   Learning adjustments: {adjustments}")
            
            print("\n✅ All tests completed!")
        
        else:
            print(f"\n🚀 Enhanced RAG System running on http://localhost:{args.port}")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    finally:
        enhanced_rag.stop()


if __name__ == "__main__":
    main()