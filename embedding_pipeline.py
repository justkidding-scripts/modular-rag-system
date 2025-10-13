#!/usr/bin/env python3
"""
Embedding Generation Pipeline for RAG System
Processes keystroke data, OCR text, and other content into semantic embeddings
"""

import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import hashlib
import numpy as np
from collections import defaultdict, deque

# Embedding models - fallback chain
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Import our RAG components
from ollama_rag_system import RAGSystem, RAGDocument


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    content: str
    content_type: str  # 'keystroke', 'ocr', 'document', 'query'
    metadata: Dict[str, Any]
    priority: int = 1
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    request: EmbeddingRequest
    embedding: List[float]
    model_used: str
    processing_time: float
    content_hash: str
    chunk_info: Optional[Dict] = None
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.request.content.encode()).hexdigest()


class TextChunker:
    """Handles text chunking for optimal embedding generation"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger("TextChunker")
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception as e:
                self.logger.warning(f"Could not initialize tiktoken: {e}")
    
    def chunk_text(self, text: str, content_type: str = 'document') -> List[Tuple[str, Dict]]:
        """Chunk text into optimal sizes for embedding"""
        if not text.strip():
            return []
        
        # Different strategies based on content type
        if content_type == 'keystroke':
            return self._chunk_keystroke_text(text)
        elif content_type == 'ocr':
            return self._chunk_ocr_text(text)
        elif content_type == 'query':
            return [(text, {'chunk_index': 0, 'total_chunks': 1})]  # Don't chunk queries
        else:
            return self._chunk_document_text(text)
    
    def _chunk_keystroke_text(self, text: str) -> List[Tuple[str, Dict]]:
        """Chunk keystroke text - usually keep sentences together"""
        # Split by sentences first
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = self._get_token_count(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, {
                    'chunk_index': len(chunks),
                    'token_count': current_length,
                    'sentence_count': len(current_chunk)
                }))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self._calculate_overlap_sentences(current_chunk):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(self._get_token_count(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, {
                'chunk_index': len(chunks),
                'token_count': current_length,
                'sentence_count': len(current_chunk)
            }))
        
        # Add total chunks info
        for _, metadata in chunks:
            metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_ocr_text(self, text: str) -> List[Tuple[str, Dict]]:
        """Chunk OCR text - preserve visual layout hints"""
        # Split by lines first to preserve layout
        lines = text.split('\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            if not line.strip():
                continue
            
            line_length = self._get_token_count(line)
            
            # If adding this line would exceed chunk size
            if current_length + line_length > self.chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append((chunk_text, {
                    'chunk_index': len(chunks),
                    'token_count': current_length,
                    'line_count': len(current_chunk),
                    'content_type': 'ocr_visual'
                }))
                
                # Start new chunk
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append((chunk_text, {
                'chunk_index': len(chunks),
                'token_count': current_length,
                'line_count': len(current_chunk),
                'content_type': 'ocr_visual'
            }))
        
        # Add total chunks info
        for _, metadata in chunks:
            metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_document_text(self, text: str) -> List[Tuple[str, Dict]]:
        """Chunk general document text"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            paragraph_length = self._get_token_count(paragraph)
            
            # If paragraph itself is too long, split it
            if paragraph_length > self.chunk_size:
                # Finalize current chunk if any
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append((chunk_text, {
                        'chunk_index': len(chunks),
                        'token_count': current_length,
                        'paragraph_count': len(current_chunk)
                    }))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = self._split_sentences(paragraph)
                for sentence_chunk, metadata in self._chunk_sentences(sentences):
                    chunks.append((sentence_chunk, {
                        **metadata,
                        'chunk_index': len(chunks),
                        'from_long_paragraph': True
                    }))
            
            # If adding this paragraph would exceed chunk size
            elif current_length + paragraph_length > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append((chunk_text, {
                    'chunk_index': len(chunks),
                    'token_count': current_length,
                    'paragraph_count': len(current_chunk)
                }))
                
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((chunk_text, {
                'chunk_index': len(chunks),
                'token_count': current_length,
                'paragraph_count': len(current_chunk)
            }))
        
        # Add total chunks info
        for _, metadata in chunks:
            metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_sentences(self, sentences: List[str]) -> List[Tuple[str, Dict]]:
        """Chunk a list of sentences"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = self._get_token_count(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, {
                    'token_count': current_length,
                    'sentence_count': len(current_chunk)
                }))
                
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, {
                'token_count': current_length,
                'sentence_count': len(current_chunk)
            }))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting - can be improved with NLTK if available
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_token_count(self, text: str) -> int:
        """Get approximate token count"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback: rough approximation
        return len(text.split()) * 1.3  # ~1.3 tokens per word on average
    
    def _calculate_overlap_sentences(self, sentences: List[str]) -> int:
        """Calculate how many sentences to include in overlap"""
        if len(sentences) <= 2:
            return 1
        
        # Aim for roughly 'overlap' tokens
        overlap_sentences = 0
        overlap_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self._get_token_count(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_tokens += sentence_tokens
                overlap_sentences += 1
            else:
                break
        
        return max(1, overlap_sentences)


class OllamaEmbedder:
    """Embedding generator using Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.logger = logging.getLogger("OllamaEmbedder")
        
        # Test connection
        self.available = self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if Ollama is available"""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Ollama not available: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Tuple[List[float], Dict]:
        """Generate embedding using Ollama"""
        if not self.available:
            raise RuntimeError("Ollama not available")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get('embedding', [])
            
            if not embedding:
                raise ValueError("Empty embedding received")
            
            processing_time = time.time() - start_time
            
            return embedding, {
                'model': self.model,
                'processing_time': processing_time,
                'embedding_dim': len(embedding),
                'backend': 'ollama'
            }
            
        except Exception as e:
            self.logger.error(f"Ollama embedding error: {e}")
            raise


class SentenceTransformersEmbedder:
    """Embedding generator using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger("SentenceTransformersEmbedder")
        
        self.available = self._load_model()
    
    def _load_model(self) -> bool:
        """Load the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded sentence transformer model: {self.model_name}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load sentence transformer model: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Tuple[List[float], Dict]:
        """Generate embedding using Sentence Transformers"""
        if not self.available or self.model is None:
            raise RuntimeError("Sentence Transformers not available")
        
        start_time = time.time()
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding_list = embedding.tolist()
            
            processing_time = time.time() - start_time
            
            return embedding_list, {
                'model': self.model_name,
                'processing_time': processing_time,
                'embedding_dim': len(embedding_list),
                'backend': 'sentence_transformers'
            }
            
        except Exception as e:
            self.logger.error(f"Sentence transformer embedding error: {e}")
            raise


class FallbackEmbedder:
    """Simple fallback embedder using basic text features"""
    
    def __init__(self):
        self.logger = logging.getLogger("FallbackEmbedder")
        self.available = True
    
    def generate_embedding(self, text: str) -> Tuple[List[float], Dict]:
        """Generate basic feature-based embedding"""
        start_time = time.time()
        
        try:
            # Extract basic text features
            features = self._extract_features(text)
            
            processing_time = time.time() - start_time
            
            return features, {
                'model': 'basic_features',
                'processing_time': processing_time,
                'embedding_dim': len(features),
                'backend': 'fallback'
            }
            
        except Exception as e:
            self.logger.error(f"Fallback embedding error: {e}")
            raise
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract basic text features as embedding"""
        import re
        from collections import Counter
        
        # Text statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Character frequency
        char_freq = Counter(text.lower())
        
        # Word length statistics
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # Basic features (384 dimensions to match common models)
        features = []
        
        # Length features (normalized)
        features.extend([
            char_count / 1000.0,  # Normalize character count
            word_count / 100.0,   # Normalize word count
            sentence_count / 10.0, # Normalize sentence count
            avg_word_length / 10.0 # Normalize average word length
        ])
        
        # Character frequency features (top 26 letters + digits + special)
        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'
        for char in alphabet:
            features.append(char_freq.get(char, 0) / max(char_count, 1))
        
        # Pad to 384 dimensions with zeros
        while len(features) < 384:
            features.append(0.0)
        
        return features[:384]  # Ensure exactly 384 dimensions


class EmbeddingPipeline:
    """Main embedding generation pipeline"""
    
    def __init__(self, storage_path: Path, chunk_size: int = 512):
        self.storage_path = Path(storage_path)
        self.chunk_size = chunk_size
        self.logger = logging.getLogger("EmbeddingPipeline")
        
        # Initialize components
        self.text_chunker = TextChunker(chunk_size=chunk_size)
        
        # Initialize embedders in priority order
        self.embedders = []
        
        # Try Ollama first
        ollama_embedder = OllamaEmbedder()
        if ollama_embedder.available:
            self.embedders.append(ollama_embedder)
            self.logger.info("✅ Ollama embedder available")
        
        # Try Sentence Transformers
        st_embedder = SentenceTransformersEmbedder()
        if st_embedder.available:
            self.embedders.append(st_embedder)
            self.logger.info("✅ Sentence Transformers embedder available")
        
        # Fallback embedder
        fallback_embedder = FallbackEmbedder()
        self.embedders.append(fallback_embedder)
        self.logger.info("✅ Fallback embedder available")
        
        # Processing queue
        self.embedding_queue = deque()
        self.processing_stats = {
            'total_processed': 0,
            'total_chunks_generated': 0,
            'processing_times': deque(maxlen=100),
            'model_usage': defaultdict(int)
        }
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_max_size = 1000
        
        print(f"✅ Embedding Pipeline initialized with {len(self.embedders)} embedders")
    
    def process_content(self, content: str, content_type: str, metadata: Dict[str, Any] = None) -> List[EmbeddingResult]:
        """Process content into embeddings"""
        if not content.strip():
            return []
        
        metadata = metadata or {}
        
        # Create embedding request
        request = EmbeddingRequest(
            content=content,
            content_type=content_type,
            metadata=metadata
        )
        
        return self._process_embedding_request(request)
    
    def _process_embedding_request(self, request: EmbeddingRequest) -> List[EmbeddingResult]:
        """Process a single embedding request"""
        results = []
        
        # Chunk the content
        chunks = self.text_chunker.chunk_text(request.content, request.content_type)
        
        if not chunks:
            return results
        
        # Process each chunk
        for chunk_text, chunk_metadata in chunks:
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            # Check cache first
            if chunk_hash in self.embedding_cache:
                cached_result = self.embedding_cache[chunk_hash]
                # Update metadata but keep cached embedding
                result = EmbeddingResult(
                    request=EmbeddingRequest(
                        content=chunk_text,
                        content_type=request.content_type,
                        metadata={**request.metadata, **chunk_metadata},
                        timestamp=request.timestamp
                    ),
                    embedding=cached_result['embedding'],
                    model_used=cached_result['model_used'],
                    processing_time=cached_result['processing_time'],
                    content_hash=chunk_hash,
                    chunk_info=chunk_metadata
                )
                results.append(result)
                continue
            
            # Generate new embedding
            try:
                embedding, embedding_metadata = self._generate_embedding(chunk_text)
                
                result = EmbeddingResult(
                    request=EmbeddingRequest(
                        content=chunk_text,
                        content_type=request.content_type,
                        metadata={**request.metadata, **chunk_metadata},
                        timestamp=request.timestamp
                    ),
                    embedding=embedding,
                    model_used=embedding_metadata['model'],
                    processing_time=embedding_metadata['processing_time'],
                    content_hash=chunk_hash,
                    chunk_info=chunk_metadata
                )
                
                results.append(result)
                
                # Cache the result
                self._cache_embedding(chunk_hash, embedding, embedding_metadata)
                
                # Update stats
                self._update_stats(embedding_metadata)
                
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for chunk: {e}")
                continue
        
        return results
    
    def _generate_embedding(self, text: str) -> Tuple[List[float], Dict]:
        """Generate embedding using available embedders"""
        last_error = None
        
        for embedder in self.embedders:
            try:
                return embedder.generate_embedding(text)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Embedder {type(embedder).__name__} failed: {e}")
                continue
        
        # If all embedders failed
        raise RuntimeError(f"All embedders failed. Last error: {last_error}")
    
    def _cache_embedding(self, content_hash: str, embedding: List[float], metadata: Dict):
        """Cache embedding result"""
        self.embedding_cache[content_hash] = {
            'embedding': embedding,
            'model_used': metadata['model'],
            'processing_time': metadata['processing_time'],
            'cached_at': time.time()
        }
        
        # Cleanup old cache entries
        if len(self.embedding_cache) > self.cache_max_size:
            # Remove oldest entries
            sorted_entries = sorted(
                self.embedding_cache.items(),
                key=lambda x: x[1]['cached_at']
            )
            # Keep newest 80% of cache
            keep_count = int(self.cache_max_size * 0.8)
            self.embedding_cache = dict(sorted_entries[-keep_count:])
    
    def _update_stats(self, embedding_metadata: Dict):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['total_chunks_generated'] += 1
        self.processing_stats['processing_times'].append(embedding_metadata['processing_time'])
        self.processing_stats['model_usage'][embedding_metadata['model']] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        processing_times = list(self.processing_stats['processing_times'])
        
        return {
            'total_processed': self.processing_stats['total_processed'],
            'total_chunks_generated': self.processing_stats['total_chunks_generated'],
            'cache_size': len(self.embedding_cache),
            'cache_hit_ratio': self._calculate_cache_hit_ratio(),
            'available_embedders': len(self.embedders),
            'embedder_info': [type(e).__name__ for e in self.embedders],
            'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'model_usage': dict(self.processing_stats['model_usage']),
            'chunk_size': self.chunk_size
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (placeholder - would need actual tracking)"""
        # This is a simplified version - real implementation would track hits vs misses
        return 0.75 if len(self.embedding_cache) > 10 else 0.0
    
    def batch_process(self, content_list: List[Tuple[str, str, Dict]], batch_size: int = 10) -> List[EmbeddingResult]:
        """Process multiple content items in batch"""
        all_results = []
        
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]
            batch_results = []
            
            for content, content_type, metadata in batch:
                results = self.process_content(content, content_type, metadata)
                batch_results.extend(results)
            
            all_results.extend(batch_results)
            
            # Small delay between batches to prevent overwhelming the system
            if i + batch_size < len(content_list):
                time.sleep(0.1)
        
        return all_results
    
    def create_rag_documents(self, embedding_results: List[EmbeddingResult]) -> List[RAGDocument]:
        """Convert embedding results to RAG documents"""
        rag_documents = []
        
        for result in embedding_results:
            # Create RAG document
            rag_doc = RAGDocument(
                content=result.request.content,
                source=result.request.content_type,
                timestamp=result.request.timestamp,
                metadata={
                    **result.request.metadata,
                    'embedding_model': result.model_used,
                    'processing_time': result.processing_time,
                    'content_hash': result.content_hash,
                    'chunk_info': result.chunk_info
                },
                embedding=result.embedding
            )
            
            rag_documents.append(rag_doc)
        
        return rag_documents


class KeystrokeEmbeddingProcessor:
    """Specialized processor for keystroke data"""
    
    def __init__(self, embedding_pipeline: EmbeddingPipeline, rag_system: RAGSystem):
        self.embedding_pipeline = embedding_pipeline
        self.rag_system = rag_system
        self.logger = logging.getLogger("KeystrokeEmbeddingProcessor")
        
        # Buffer for keystroke data
        self.keystroke_buffer = deque(maxlen=100)
        self.batch_size = 5
        self.batch_timeout = 30  # seconds
        self.last_batch_time = time.time()
        
        # Processing thread
        self.processing_thread = None
        self.stop_processing = False
    
    def add_keystroke_data(self, content: str, metadata: Dict[str, Any]):
        """Add keystroke data to processing queue"""
        self.keystroke_buffer.append((content, metadata, time.time()))
        
        # Trigger batch processing if buffer is full or timeout reached
        if (len(self.keystroke_buffer) >= self.batch_size or 
            time.time() - self.last_batch_time > self.batch_timeout):
            
            self._process_batch()
    
    def _process_batch(self):
        """Process current batch of keystroke data"""
        if not self.keystroke_buffer:
            return
        
        try:
            # Prepare batch data
            batch_data = []
            
            while self.keystroke_buffer:
                content, metadata, timestamp = self.keystroke_buffer.popleft()
                
                # Enhanced metadata for keystroke embeddings
                enhanced_metadata = {
                    **metadata,
                    'processed_at': time.time(),
                    'original_timestamp': timestamp,
                    'batch_processing': True
                }
                
                batch_data.append((content, 'keystroke', enhanced_metadata))
            
            if not batch_data:
                return
            
            # Generate embeddings
            embedding_results = self.embedding_pipeline.batch_process(batch_data)
            
            # Convert to RAG documents
            rag_documents = self.embedding_pipeline.create_rag_documents(embedding_results)
            
            # Add to RAG system
            if rag_documents:
                self.rag_system.add_documents(rag_documents)
                self.logger.info(f"Processed and added {len(rag_documents)} keystroke documents to RAG system")
            
            self.last_batch_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
    
    def start_background_processing(self):
        """Start background processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._background_processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("Started background keystroke processing")
    
    def stop_background_processing(self):
        """Stop background processing"""
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Process any remaining data
        self._process_batch()
        self.logger.info("Stopped background keystroke processing")
    
    def _background_processing_loop(self):
        """Background processing loop"""
        while not self.stop_processing:
            try:
                # Check if it's time to process a batch
                if (len(self.keystroke_buffer) > 0 and 
                    time.time() - self.last_batch_time > self.batch_timeout):
                    
                    self._process_batch()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Background processing error: {e}")
                time.sleep(10)  # Wait longer on error


def main():
    """Main function for testing embedding pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Generation Pipeline")
    parser.add_argument("--storage", type=Path, default="./rag_storage", help="Storage path")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for embeddings")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EmbeddingPipeline(args.storage, chunk_size=args.chunk_size)
    
    if args.test:
        print("🧪 Running embedding pipeline tests...")
        
        # Test different content types
        test_data = [
            ("This is a test keystroke input from the user typing in VSCode.", "keystroke", {"application": "vscode"}),
            ("Screen text detected via OCR\nMultiple lines\nWith layout preserved", "ocr", {"confidence": 0.95}),
            ("This is a longer document that should be chunked into multiple pieces. " * 20, "document", {"source": "test"}),
            ("What is machine learning?", "query", {})
        ]
        
        for content, content_type, metadata in test_data:
            print(f"\n📝 Processing {content_type}: {content[:50]}...")
            
            results = pipeline.process_content(content, content_type, metadata)
            
            print(f"   Generated {len(results)} embeddings")
            for i, result in enumerate(results):
                print(f"   Chunk {i+1}: {len(result.embedding)} dimensions, {result.model_used}, {result.processing_time:.3f}s")
        
        # Show stats
        stats = pipeline.get_stats()
        print(f"\n📊 Pipeline Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    else:
        print("Embedding pipeline initialized")
        print("Use --test to run test mode")


if __name__ == "__main__":
    main()