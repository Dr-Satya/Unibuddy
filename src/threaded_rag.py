import hashlib
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from src.config import settings
from src.database import db_manager

@dataclass
class RetrievalResult:
    content: str
    score: float
    source: str
    extra_metadata: Optional[Dict[str, Any]] = None

class ThreadedRAGSystem:
    """Enhanced multi-threaded Retrieval-Augmented Generation system for university data."""
    
    def __init__(self, max_workers: int = 4):
        self.embedding_model = None
        self.vector_db = None
        self.text_splitter = None
        self.index_to_doc_map = {}
        self.is_initialized = False
        self.max_workers = max_workers
        self.processing_lock = threading.Lock()
        
        # Batch processing settings
        self.batch_size = 32  # Process chunks in batches for efficiency
        self.embedding_cache = {}  # Cache embeddings to avoid recomputation
        
    def _initialize(self):
        """Initialize RAG system components."""
        if self.is_initialized:
            return
        
        try:
            print("🤖 Initializing Threaded RAG system...")
            
            # Initialize embedding model with CPU optimization
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Optimize for CPU inference
            if hasattr(self.embedding_model, '_modules'):
                for module in self.embedding_model._modules.values():
                    if hasattr(module, 'eval'):
                        module.eval()
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Initialize or load FAISS index
            self._load_or_create_index()
            
            self.is_initialized = True
            print("✅ Threaded RAG system initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            self.is_initialized = False
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        index_path = os.path.join(settings.VECTOR_DB_PATH, "index.faiss")
        metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing index
            try:
                self.vector_db = faiss.read_index(index_path)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.index_to_doc_map = json.load(f)
                print(f"📚 Loaded existing FAISS index with {self.vector_db.ntotal} vectors")
            except Exception as e:
                print(f"⚠️ Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Create directory if it doesn't exist
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # Create new FAISS index (using L2 distance)
        embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.vector_db = faiss.IndexFlatL2(embedding_dim)
        self.index_to_doc_map = {}
        
        print("🆕 Created new FAISS index")
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        if not self.vector_db:
            return
            
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        index_path = os.path.join(settings.VECTOR_DB_PATH, "index.faiss")
        metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        
        faiss.write_index(self.vector_db, index_path)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_to_doc_map, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved FAISS index with {self.vector_db.ntotal} vectors")
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        try:
            # Use batch processing for efficiency
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=min(self.batch_size, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            print(f"⚠️ Error generating embeddings: {e}")
            return np.array([])
    
    def _process_document_batch(self, doc_batch: List[Dict[str, Any]], batch_id: int) -> List[Dict[str, Any]]:
        """Process a batch of documents and return chunk data."""
        print(f"🔄 [Batch {batch_id}] Processing {len(doc_batch)} documents...")
        batch_chunks = []
        
        for doc in doc_batch:
            content = doc.get('content', '')
            if not content:
                continue
            
            # Create document ID
            doc_id = doc.get('id') or hashlib.sha256(content.encode()).hexdigest()
            
            # Split document into chunks
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                chunk_data = {
                    'chunk': chunk,
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'metadata': {
                        'content': chunk,
                        'source': doc.get('url', 'Unknown'),
                        'title': doc.get('title', 'Untitled'),
                        'content_type': doc.get('content_type', 'general'),
                        'chunk_index': i,
                        'doc_id': doc_id,
                        'scraped_at': doc.get('scraped_at', ''),
                        'metadata': doc.get('metadata', {}),
                        'batch_id': batch_id
                    }
                }
                batch_chunks.append(chunk_data)
        
        print(f"✅ [Batch {batch_id}] Generated {len(batch_chunks)} chunks")
        return batch_chunks
    
    def add_documents_threaded(self, documents: List[Dict[str, Any]], progress_callback: Optional[Callable[[int, int], None]] = None):
        """Add documents to the RAG system using threading for faster processing."""
        self._initialize()
        
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")
        
        print(f"🚀 Starting threaded document processing for {len(documents)} documents...")
        start_time = time.time()
        
        # Split documents into batches for processing
        doc_batches = []
        batch_size = max(1, len(documents) // self.max_workers)
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            doc_batches.append((batch, i // batch_size))
        
        all_chunks = []
        
        # Process document batches concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="DocProcessor") as executor:
            # Submit batch processing tasks
            futures = {
                executor.submit(self._process_document_batch, batch, batch_id): batch_id
                for batch, batch_id in doc_batches
            }
            
            completed_batches = 0
            total_batches = len(futures)
            
            # Collect results
            for future in as_completed(futures):
                batch_id = futures[future]
                try:
                    batch_chunks = future.result(timeout=120)
                    all_chunks.extend(batch_chunks)
                    completed_batches += 1
                    
                    if progress_callback:
                        progress_callback(completed_batches, total_batches)
                    
                    print(f"📊 [{completed_batches}/{total_batches}] Batch {batch_id} completed")
                    
                except Exception as e:
                    print(f"❌ Error processing batch {batch_id}: {e}")
        
        print(f"🔄 Generated {len(all_chunks)} chunks, now creating embeddings...")
        
        # Process embeddings in batches
        self._add_chunks_with_embeddings(all_chunks)
        
        # Save the updated index
        self._save_index()
        
        elapsed_time = time.time() - start_time
        print(f"🎉 Threaded document processing completed in {elapsed_time:.2f} seconds!")
        print(f"📚 Added {len(all_chunks)} chunks to RAG system")
    
    def _add_chunks_with_embeddings(self, chunks: List[Dict[str, Any]]):
        """Add chunks to vector store with batch embedding generation."""
        if not chunks:
            return
        
        print(f"🧠 Generating embeddings for {len(chunks)} chunks in batches...")
        
        # Group chunks into embedding batches
        embedding_batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            embedding_batches.append(batch)
        
        with ThreadPoolExecutor(max_workers=min(4, len(embedding_batches)), thread_name_prefix="Embedder") as executor:
            futures = {}
            
            # Submit embedding generation tasks
            for batch_idx, chunk_batch in enumerate(embedding_batches):
                texts = [chunk['chunk'] for chunk in chunk_batch]
                future = executor.submit(self._generate_embeddings_batch, texts)
                futures[future] = (batch_idx, chunk_batch)
            
            # Process results as they complete
            for future in as_completed(futures):
                batch_idx, chunk_batch = futures[future]
                
                try:
                    embeddings = future.result(timeout=60)
                    if embeddings.size > 0:
                        self._add_embeddings_to_index(chunk_batch, embeddings, batch_idx)
                    else:
                        print(f"⚠️ Empty embeddings for batch {batch_idx}")
                        
                except Exception as e:
                    print(f"❌ Error processing embedding batch {batch_idx}: {e}")
    
    def _add_embeddings_to_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, batch_idx: int):
        """Add embeddings and metadata to FAISS index."""
        with self.processing_lock:
            for chunk_data, embedding in zip(chunks, embeddings):
                # Add to FAISS index
                current_index = self.vector_db.ntotal
                self.vector_db.add(embedding.reshape(1, -1))
                
                # Store metadata
                self.index_to_doc_map[str(current_index)] = chunk_data['metadata']
                
                # Store in database as well
                chunk_db_data = {
                    'id': f"{chunk_data['doc_id']}_chunk_{chunk_data['chunk_index']}",
                    'university_data_id': chunk_data['doc_id'],
                    'chunk_text': chunk_data['chunk'],
                    'chunk_index': chunk_data['chunk_index'],
                    'vector_id': str(current_index)
                }
                
                try:
                    db_manager.store_document_chunk(chunk_db_data)
                except Exception as e:
                    print(f"⚠️ Warning: Could not store chunk in database: {e}")
            
            print(f"✅ Added batch {batch_idx} with {len(chunks)} chunks to index")
    
    def search(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Search for relevant documents."""
        self._initialize()
        
        if not self.is_initialized or not self.vector_db:
            return []
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        if self.vector_db.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search FAISS index
            scores, indices = self.vector_db.search(
                query_embedding.reshape(1, -1), 
                min(top_k, self.vector_db.ntotal)
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                doc_data = self.index_to_doc_map.get(str(idx))
                if not doc_data:
                    continue
                
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + score)
                
                result = RetrievalResult(
                    content=doc_data['content'],
                    score=similarity_score,
                    source=doc_data['source'],
                    extra_metadata={
                        'title': doc_data.get('title', ''),
                        'content_type': doc_data.get('content_type', ''),
                        'chunk_index': doc_data.get('chunk_index', 0),
                        'doc_id': doc_data.get('doc_id', ''),
                        'scraped_at': doc_data.get('scraped_at', ''),
                        'raw_score': float(score),
                        'batch_id': doc_data.get('batch_id', 'unknown')
                    }
                )
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for a query, formatted for RAG."""
        results = self.search(query)
        
        if not results:
            return "No relevant information found in the university database."
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result.content.strip()
            if not content:
                continue
            
            # Add source information
            source_info = f"[Source: {result.extra_metadata.get('title', 'Unknown') if result.extra_metadata else 'Unknown'}]"
            full_content = f"{source_info}\n{content}"
            
            if current_length + len(full_content) > max_context_length:
                break
            
            context_parts.append(full_content)
            current_length += len(full_content)
        
        context = "\n\n---\n\n".join(context_parts)
        
        return f"""Relevant information from GD Goenka University database:

{context}

Please use this information to answer the user's question about the university."""
    
    def update_from_database(self):
        """Update RAG system with latest data from database using threading."""
        try:
            print("🔄 Updating RAG system from database...")
            university_data = db_manager.get_university_data()
            
            documents = []
            for data in university_data:
                doc = {
                    'id': data.id,
                    'content': data.content,
                    'url': data.url,
                    'title': data.title,
                    'content_type': data.content_type,
                    'scraped_at': data.scraped_at.isoformat() if data.scraped_at else '',
                    'metadata': data.extra_metadata or {}
                }
                documents.append(doc)
            
            if documents:
                # Clear existing index and rebuild with threading
                self._create_new_index()
                self.add_documents_threaded(documents)
                print(f"🔄 Updated RAG system with {len(documents)} documents using threading")
            else:
                print("⚠️ No university data found in database")
                
        except Exception as e:
            print(f"❌ Error updating RAG system from database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        self._initialize()
        
        stats = {
            'initialized': self.is_initialized,
            'total_vectors': self.vector_db.ntotal if self.vector_db else 0,
            'total_documents': len(set(
                doc.get('doc_id', '') for doc in self.index_to_doc_map.values()
            )) if self.index_to_doc_map else 0,
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': settings.CHUNK_SIZE,
            'chunk_overlap': settings.CHUNK_OVERLAP,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'threading_enabled': True
        }
        
        return stats
    
    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self.embedding_cache.clear()
        print("🧹 Cleared embedding cache")

# Global threaded RAG system instance
threaded_rag_system = ThreadedRAGSystem(max_workers=4)