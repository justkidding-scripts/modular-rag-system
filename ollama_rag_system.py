#!/usr/bin/env python3
"""
Ollama RAG (Retrieval-Augmented Generation) System
Provides contextual memory and real-time data integration for enhanced AI responses
"""

import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import sqlite3
import numpy as np
from collections import defaultdict, deque
import pickle
import gzip

# Try to import advanced dependencies with fallbacks
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available - using SQLite fallback")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available - using Ollama embeddings")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available - using basic similarity search")


@dataclass
class RAGDocument:
    """Represents a document in the RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: float = 0.0
    source: str = "unknown"
    importance_score: float = 1.0
    access_count: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.id:
            self.id = self.generate_id()
    
    def generate_id(self) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        time_hash = hashlib.md5(str(self.timestamp).encode()).hexdigest()[:4]
        return f"{self.source}_{content_hash}_{time_hash}"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RAGQuery:
    """Represents a query to the RAG system"""
    query_text: str
    context: Dict[str, Any]
    max_results: int = 5
    similarity_threshold: float = 0.7
    time_weight: float = 0.1
    source_filters: Optional[List[str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class RAGResult:
    """Represents a result from RAG retrieval"""
    documents: List[RAGDocument]
    similarities: List[float]
    query_embedding: Optional[List[float]] = None
    retrieval_time: float = 0.0
    total_documents_searched: int = 0


class EmbeddingGenerator:
    """Handles text embedding generation with multiple backends"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_ollama: bool = True):
        self.use_ollama = use_ollama
        self.embedding_model = embedding_model
        self.sentence_transformer = None
        self.embedding_cache = {}
        
        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE and not use_ollama:
            try:
                self.sentence_transformer = SentenceTransformer(embedding_model)
                print(f"✅ SentenceTransformer loaded: {embedding_model}")
            except Exception as e:
                print(f"❌ Failed to load SentenceTransformer: {e}")
                self.sentence_transformer = None
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for text"""
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = None
        
        # Try Ollama embeddings first
        if self.use_ollama:
            embedding = self._generate_ollama_embedding(text)
        
        # Fallback to sentence transformers
        if embedding is None and self.sentence_transformer:
            embedding = self._generate_sentence_transformer_embedding(text)
        
        # Final fallback to simple text hashing
        if embedding is None:
            embedding = self._generate_simple_embedding(text)
        
        if use_cache and embedding:
            self.embedding_cache[text] = embedding
        
        return embedding or []
    
    def _generate_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "llama3.2:3b",  # or your preferred model
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('embedding', [])
        except Exception as e:
            print(f"Ollama embedding error: {e}")
        
        return None
    
    def _generate_sentence_transformer_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using SentenceTransformers"""
        try:
            if self.sentence_transformer:
                embedding = self.sentence_transformer.encode(text)
                return embedding.tolist()
        except Exception as e:
            print(f"SentenceTransformer embedding error: {e}")
        
        return None
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding using text features"""
        # Basic text features as embedding
        words = text.lower().split()
        char_counts = defaultdict(int)
        
        for char in text.lower():
            if char.isalpha():
                char_counts[char] += 1
        
        # Create 128-dimensional embedding
        embedding = [0.0] * 128
        
        # Word count features
        embedding[0] = min(len(words) / 100.0, 1.0)
        embedding[1] = min(len(text) / 1000.0, 1.0)
        
        # Character frequency features
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            if i < 26:
                embedding[i + 2] = char_counts.get(char, 0) / max(len(text), 1)
        
        # Simple hash features for remaining dimensions
        text_hash = hash(text)
        for i in range(28, 128):
            embedding[i] = ((text_hash >> (i % 32)) & 1) * 0.1
        
        return embedding
    
    def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if not emb1 or not emb2 or len(emb1) != len(emb2):
            return 0.0
        
        try:
            # Convert to numpy arrays
            a = np.array(emb1)
            b = np.array(emb2)
            
            # Cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0


class RAGVectorStore:
    """Vector storage and retrieval system for RAG"""
    
    def __init__(self, storage_path: Path, use_chromadb: bool = True):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize storage backend
        if self.use_chromadb:
            self._init_chromadb()
        else:
            self._init_sqlite_fallback()
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.document_mapping = {}
        
        if FAISS_AVAILABLE:
            self._init_faiss_index()
        
        print(f"✅ RAG Vector Store initialized: {storage_path}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.storage_path / "chromadb"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name="rag_documents",
                metadata={"description": "RAG document storage"}
            )
            
            print("✅ ChromaDB initialized")
            
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            self.use_chromadb = False
            self._init_sqlite_fallback()
    
    def _init_sqlite_fallback(self):
        """Initialize SQLite fallback storage"""
        self.db_path = self.storage_path / "rag_documents.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    timestamp REAL,
                    source TEXT,
                    importance_score REAL,
                    access_count INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON documents(timestamp);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON documents(source);
            """)
        
        print("✅ SQLite fallback storage initialized")
    
    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            # Start with a basic index, will rebuild when documents are added
            self.faiss_index = faiss.IndexFlatIP(128)  # Inner product (cosine with normalized vectors)
            print("✅ FAISS index initialized")
        except Exception as e:
            print(f"❌ FAISS initialization failed: {e}")
    
    def add_document(self, document: RAGDocument) -> bool:
        """Add document to vector store"""
        try:
            # Generate embedding if not provided
            if not document.embedding:
                document.embedding = self.embedding_generator.generate_embedding(document.content)
            
            if self.use_chromadb:
                return self._add_document_chromadb(document)
            else:
                return self._add_document_sqlite(document)
                
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def _add_document_chromadb(self, document: RAGDocument) -> bool:
        """Add document to ChromaDB"""
        try:
            self.collection.upsert(
                ids=[document.id],
                documents=[document.content],
                embeddings=[document.embedding] if document.embedding else None,
                metadatas=[{
                    **document.metadata,
                    'timestamp': document.timestamp,
                    'source': document.source,
                    'importance_score': document.importance_score,
                    'access_count': document.access_count
                }]
            )
            
            self._update_faiss_index(document)
            return True
            
        except Exception as e:
            print(f"ChromaDB add error: {e}")
            return False
    
    def _add_document_sqlite(self, document: RAGDocument) -> bool:
        """Add document to SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, content, metadata, embedding, timestamp, source, importance_score, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.id,
                    document.content,
                    json.dumps(document.metadata),
                    pickle.dumps(document.embedding) if document.embedding else None,
                    document.timestamp,
                    document.source,
                    document.importance_score,
                    document.access_count
                ))
            
            self._update_faiss_index(document)
            return True
            
        except Exception as e:
            print(f"SQLite add error: {e}")
            return False
    
    def _update_faiss_index(self, document: RAGDocument):
        """Update FAISS index with new document"""
        if not FAISS_AVAILABLE or not document.embedding:
            return
        
        try:
            # Normalize embedding for cosine similarity
            embedding = np.array(document.embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            self.faiss_index.add(embedding.reshape(1, -1))
            self.document_mapping[self.faiss_index.ntotal - 1] = document.id
            
        except Exception as e:
            print(f"FAISS update error: {e}")
    
    def search(self, query: RAGQuery) -> RAGResult:
        """Search for similar documents"""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query.query_text)
        
        if self.use_chromadb:
            results = self._search_chromadb(query, query_embedding)
        else:
            results = self._search_sqlite(query, query_embedding)
        
        results.query_embedding = query_embedding
        results.retrieval_time = time.time() - start_time
        
        return results
    
    def _search_chromadb(self, query: RAGQuery, query_embedding: List[float]) -> RAGResult:
        """Search using ChromaDB"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if query.metadata_filters:
                where_clause.update(query.metadata_filters)
            
            if query.source_filters:
                where_clause["source"] = {"$in": query.source_filters}
            
            results = self.collection.query(
                query_embeddings=[query_embedding] if query_embedding else None,
                query_texts=[query.query_text] if not query_embedding else None,
                n_results=query.max_results,
                where=where_clause if where_clause else None
            )
            
            documents = []
            similarities = []
            
            for i in range(len(results['ids'][0])):
                doc = RAGDocument(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    embedding=results['embeddings'][0][i] if results['embeddings'] else None,
                    timestamp=results['metadatas'][0][i].get('timestamp', 0),
                    source=results['metadatas'][0][i].get('source', 'unknown'),
                    importance_score=results['metadatas'][0][i].get('importance_score', 1.0),
                    access_count=results['metadatas'][0][i].get('access_count', 0)
                )
                
                # Update access count
                doc.access_count += 1
                
                documents.append(doc)
                similarities.append(results['distances'][0][i] if results['distances'] else 0.8)
            
            return RAGResult(
                documents=documents,
                similarities=similarities,
                total_documents_searched=self.collection.count()
            )
            
        except Exception as e:
            print(f"ChromaDB search error: {e}")
            return RAGResult(documents=[], similarities=[])
    
    def _search_sqlite(self, query: RAGQuery, query_embedding: List[float]) -> RAGResult:
        """Search using SQLite with manual similarity calculation"""
        documents = []
        similarities = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build SQL query
                sql = "SELECT * FROM documents"
                params = []
                
                where_conditions = []
                if query.source_filters:
                    placeholders = ','.join(['?'] * len(query.source_filters))
                    where_conditions.append(f"source IN ({placeholders})")
                    params.extend(query.source_filters)
                
                if where_conditions:
                    sql += " WHERE " + " AND ".join(where_conditions)
                
                sql += " ORDER BY timestamp DESC LIMIT 1000"  # Limit for performance
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                # Calculate similarities manually
                for row in rows:
                    doc_id, content, metadata_json, embedding_blob, timestamp, source, importance_score, access_count = row
                    
                    # Deserialize embedding
                    doc_embedding = None
                    if embedding_blob:
                        try:
                            doc_embedding = pickle.loads(embedding_blob)
                        except:
                            continue
                    
                    # Calculate similarity
                    if query_embedding and doc_embedding:
                        similarity = self.embedding_generator.calculate_similarity(query_embedding, doc_embedding)
                        
                        if similarity >= query.similarity_threshold:
                            doc = RAGDocument(
                                id=doc_id,
                                content=content,
                                metadata=json.loads(metadata_json) if metadata_json else {},
                                embedding=doc_embedding,
                                timestamp=timestamp,
                                source=source,
                                importance_score=importance_score,
                                access_count=access_count + 1
                            )
                            
                            documents.append(doc)
                            similarities.append(similarity)
                
                # Sort by similarity and limit results
                if documents:
                    sorted_results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
                    documents, similarities = zip(*sorted_results[:query.max_results])
                    documents = list(documents)
                    similarities = list(similarities)
            
            return RAGResult(
                documents=documents,
                similarities=similarities,
                total_documents_searched=len(rows) if 'rows' in locals() else 0
            )
            
        except Exception as e:
            print(f"SQLite search error: {e}")
            return RAGResult(documents=[], similarities=[])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'backend': 'chromadb' if self.use_chromadb else 'sqlite',
            'total_documents': 0,
            'sources': {},
            'embedding_cache_size': len(self.embedding_generator.embedding_cache)
        }
        
        try:
            if self.use_chromadb:
                stats['total_documents'] = self.collection.count()
                # Get source breakdown
                all_docs = self.collection.get()
                for metadata in all_docs['metadatas']:
                    source = metadata.get('source', 'unknown')
                    stats['sources'][source] = stats['sources'].get(source, 0) + 1
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM documents")
                    stats['total_documents'] = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT source, COUNT(*) FROM documents GROUP BY source")
                    stats['sources'] = dict(cursor.fetchall())
        except Exception as e:
            print(f"Stats error: {e}")
        
        return stats


class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, storage_path: Path, config: Optional[Dict] = None):
        self.storage_path = Path(storage_path)
        self.config = config or self._default_config()
        
        # Initialize components
        self.vector_store = RAGVectorStore(self.storage_path / "vectors")
        self.logger = self._setup_logging()
        
        # Document processing
        self.document_queue = deque(maxlen=1000)
        self.processing_thread = None
        self.processing_active = False
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'cache_hits': 0
        }
        
        # Context memory
        self.recent_context = deque(maxlen=100)
        
        print(f"✅ RAG System initialized: {storage_path}")
    
    def _default_config(self) -> Dict:
        """Default RAG system configuration"""
        return {
            'embedding': {
                'model': 'all-MiniLM-L6-v2',
                'use_ollama': True,
                'cache_size': 10000
            },
            'retrieval': {
                'max_results': 5,
                'similarity_threshold': 0.7,
                'time_weight': 0.1,
                'context_window': 50
            },
            'processing': {
                'batch_size': 10,
                'auto_process': True,
                'cleanup_interval': 3600  # 1 hour
            },
            'privacy': {
                'filter_sensitive': True,
                'max_retention_days': 30,
                'anonymize_data': False
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for RAG system"""
        logger = logging.getLogger("RAGSystem")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.storage_path / "rag_system.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def add_document(self, content: str, metadata: Optional[Dict] = None, 
                    source: str = "unknown", importance: float = 1.0) -> str:
        """Add document to RAG system"""
        doc = RAGDocument(
            id="",  # Will be generated
            content=content,
            metadata=metadata or {},
            source=source,
            importance_score=importance
        )
        
        if self.config['processing']['auto_process']:
            # Add to queue for background processing
            self.document_queue.append(doc)
            self._ensure_processing_thread()
        else:
            # Process immediately
            success = self.vector_store.add_document(doc)
            if success:
                self.logger.info(f"Document added: {doc.id}")
            
        return doc.id
    
    def _ensure_processing_thread(self):
        """Ensure document processing thread is running"""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(target=self._process_documents, daemon=True)
            self.processing_thread.start()
    
    def _process_documents(self):
        """Background document processing"""
        batch_size = self.config['processing']['batch_size']
        
        while self.processing_active:
            try:
                # Process documents in batches
                batch = []
                for _ in range(batch_size):
                    if self.document_queue:
                        batch.append(self.document_queue.popleft())
                    else:
                        break
                
                if batch:
                    for doc in batch:
                        success = self.vector_store.add_document(doc)
                        if success:
                            self.logger.info(f"Background processed: {doc.id}")
                
                # Sleep if no documents to process
                if not batch:
                    time.sleep(0.5)
                    
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                time.sleep(1)
    
    def query(self, query_text: str, context: Optional[Dict] = None, 
             max_results: int = None, source_filters: Optional[List[str]] = None) -> RAGResult:
        """Query the RAG system"""
        start_time = time.time()
        
        # Build query
        query = RAGQuery(
            query_text=query_text,
            context=context or {},
            max_results=max_results or self.config['retrieval']['max_results'],
            similarity_threshold=self.config['retrieval']['similarity_threshold'],
            source_filters=source_filters
        )
        
        # Add recent context to query
        if self.recent_context:
            query.context['recent_activity'] = list(self.recent_context)[-10:]
        
        # Perform search
        results = self.vector_store.search(query)
        
        # Update stats
        self.query_stats['total_queries'] += 1
        retrieval_time = time.time() - start_time
        self.query_stats['avg_retrieval_time'] = (
            (self.query_stats['avg_retrieval_time'] * (self.query_stats['total_queries'] - 1) + 
             retrieval_time) / self.query_stats['total_queries']
        )
        
        # Store context for future queries
        self.recent_context.append({
            'query': query_text,
            'timestamp': time.time(),
            'results_count': len(results.documents)
        })
        
        self.logger.info(f"Query processed: '{query_text[:50]}...' - {len(results.documents)} results in {retrieval_time:.3f}s")
        
        return results
    
    def get_enhanced_context(self, query_text: str, current_context: Dict = None) -> Dict:
        """Get enhanced context by combining current context with RAG results"""
        # Query RAG system
        rag_results = self.query(query_text, current_context)
        
        enhanced_context = current_context.copy() if current_context else {}
        
        # Add RAG results to context
        enhanced_context.update({
            'rag_documents': [doc.to_dict() for doc in rag_results.documents],
            'rag_similarities': rag_results.similarities,
            'rag_query_embedding': rag_results.query_embedding,
            'rag_retrieval_time': rag_results.retrieval_time,
            'total_documents_searched': rag_results.total_documents_searched,
            'historical_context': self._build_historical_context(rag_results),
            'context_summary': self._summarize_context(rag_results)
        })
        
        return enhanced_context
    
    def _build_historical_context(self, rag_results: RAGResult) -> Dict:
        """Build historical context from RAG results"""
        if not rag_results.documents:
            return {}
        
        # Analyze temporal patterns
        timestamps = [doc.timestamp for doc in rag_results.documents]
        sources = [doc.source for doc in rag_results.documents]
        
        historical_context = {
            'time_span': {
                'earliest': min(timestamps) if timestamps else 0,
                'latest': max(timestamps) if timestamps else 0,
                'span_hours': (max(timestamps) - min(timestamps)) / 3600 if len(timestamps) > 1 else 0
            },
            'source_distribution': {},
            'activity_patterns': self._analyze_activity_patterns(rag_results.documents),
            'content_themes': self._extract_content_themes(rag_results.documents)
        }
        
        # Count sources
        for source in sources:
            historical_context['source_distribution'][source] = historical_context['source_distribution'].get(source, 0) + 1
        
        return historical_context
    
    def _analyze_activity_patterns(self, documents: List[RAGDocument]) -> Dict:
        """Analyze activity patterns from documents"""
        patterns = {
            'peak_hours': defaultdict(int),
            'common_activities': defaultdict(int),
            'session_lengths': []
        }
        
        for doc in documents:
            # Extract hour from timestamp
            dt = datetime.fromtimestamp(doc.timestamp)
            patterns['peak_hours'][dt.hour] += 1
            
            # Extract activity type from metadata or content
            activity = doc.metadata.get('activity_type', 'unknown')
            patterns['common_activities'][activity] += 1
        
        return dict(patterns)
    
    def _extract_content_themes(self, documents: List[RAGDocument]) -> List[str]:
        """Extract common themes from document content"""
        # Simple keyword-based theme extraction
        all_content = " ".join([doc.content for doc in documents])
        words = all_content.lower().split()
        
        # Count word frequency (excluding common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'}
        
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_freq[word] += 1
        
        # Return top themes
        return [word for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    def _summarize_context(self, rag_results: RAGResult) -> str:
        """Create a summary of the context for use in prompts"""
        if not rag_results.documents:
            return "No relevant historical context found."
        
        doc_count = len(rag_results.documents)
        avg_similarity = sum(rag_results.similarities) / len(rag_results.similarities) if rag_results.similarities else 0
        
        # Get most relevant document
        most_relevant = rag_results.documents[0] if rag_results.documents else None
        
        summary = f"Found {doc_count} relevant documents (avg similarity: {avg_similarity:.2f}). "
        
        if most_relevant:
            summary += f"Most relevant: {most_relevant.content[:100]}..."
            if most_relevant.metadata:
                summary += f" (from {most_relevant.source})"
        
        return summary
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        vector_stats = self.vector_store.get_stats()
        
        stats = {
            'vector_store': vector_stats,
            'query_stats': self.query_stats,
            'processing': {
                'queue_size': len(self.document_queue),
                'processing_active': self.processing_active,
                'recent_context_size': len(self.recent_context)
            },
            'config': self.config,
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }
        
        return stats
    
    def cleanup_old_documents(self, max_age_days: int = None):
        """Clean up old documents based on retention policy"""
        max_age = max_age_days or self.config['privacy']['max_retention_days']
        cutoff_time = time.time() - (max_age * 24 * 3600)
        
        # This would need to be implemented based on the specific storage backend
        # For now, log the intent
        self.logger.info(f"Cleanup requested for documents older than {max_age} days")
    
    def export_data(self, export_path: Path) -> bool:
        """Export RAG data for backup or analysis"""
        try:
            export_data = {
                'timestamp': time.time(),
                'stats': self.get_system_stats(),
                'recent_context': list(self.recent_context)
            }
            
            with gzip.open(export_path, 'wt') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Data exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return False
    
    def shutdown(self):
        """Graceful shutdown"""
        self.processing_active = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        self.logger.info("RAG System shutdown complete")

