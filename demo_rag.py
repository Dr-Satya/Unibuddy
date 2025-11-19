#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) Demo

This script demonstrates how the university data gets "trained" into the system.
Instead of traditional model training, we use RAG which:
1. Chunks the university data
2. Creates embeddings (vectors) for each chunk
3. Stores them in a vector database
4. At query time, finds relevant chunks and uses them as context

This is more efficient and flexible than retraining the entire model.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

def load_scraped_data() -> List[Dict[str, Any]]:
    """Load the scraped university data."""
    data_sources = [
        'data/processed/scraping_test_result.json',
        'data/processed/sample_university_data.json'
    ]
    
    documents = []
    
    for source in data_sources:
        if os.path.exists(source):
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    documents.append(data)
                    print(f"✅ Loaded: {source}")
            except Exception as e:
                print(f"❌ Error loading {source}: {e}")
    
    return documents

def chunk_text_simple(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple text chunking function."""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            chunks.append(chunk.strip())
    
    return chunks

def create_mock_embeddings(text: str) -> List[float]:
    """Create mock embeddings (in reality, this would use sentence transformers)."""
    # This is just a simple hash-based mock embedding
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert to "embedding-like" vector
    embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(len(hash_hex), 128), 2)]
    
    # Pad to consistent length
    while len(embedding) < 64:
        embedding.append(0.0)
    
    return embedding[:64]

def process_university_data(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process university data for RAG system."""
    print("\\n🧠 Processing University Data for RAG System")
    print("=" * 50)
    
    processed_data = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": len(documents),
        "chunks": [],
        "vector_index": {}
    }
    
    chunk_id = 0
    
    for doc_idx, doc in enumerate(documents):
        print(f"\\n📄 Processing document {doc_idx + 1}...")
        
        # Extract content from document
        if 'content' in doc and isinstance(doc['content'], str):
            content = doc['content']
            source = doc.get('url', f'Document_{doc_idx}')
            title = doc.get('title', f'University Document {doc_idx}')
        else:
            # Handle structured data (like sample data)
            content = json.dumps(doc, indent=2)
            source = "Sample University Data"
            title = "University Information"
        
        print(f"  📝 Title: {title}")
        print(f"  📍 Source: {source}")
        print(f"  📏 Content length: {len(content)} characters")
        
        # Chunk the content
        chunks = chunk_text_simple(content)
        print(f"  🧩 Created {len(chunks)} chunks")
        
        for chunk_idx, chunk in enumerate(chunks):
            # Create mock embedding
            embedding = create_mock_embeddings(chunk)
            
            chunk_data = {
                "id": f"chunk_{chunk_id}",
                "document_id": doc_idx,
                "chunk_index": chunk_idx,
                "content": chunk,
                "source": source,
                "title": title,
                "embedding_size": len(embedding),
                "content_length": len(chunk)
            }
            
            processed_data["chunks"].append(chunk_data)
            processed_data["vector_index"][f"chunk_{chunk_id}"] = embedding
            chunk_id += 1
    
    return processed_data

def search_similar_chunks(query: str, processed_data: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for similar chunks (mock similarity search)."""
    query_embedding = create_mock_embeddings(query)
    
    similarities = []
    
    for chunk in processed_data["chunks"]:
        chunk_id = chunk["id"]
        chunk_embedding = processed_data["vector_index"][chunk_id]
        
        # Simple cosine similarity (mock)
        similarity = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
        similarities.append({
            "chunk": chunk,
            "similarity": similarity
        })
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return [item["chunk"] for item in similarities[:top_k]]

def demo_rag_search(processed_data: Dict[str, Any]):
    """Demonstrate RAG search functionality."""
    print("\\n🔍 RAG Search Demonstration")
    print("=" * 40)
    
    test_queries = [
        "What is the fee structure for B.Tech?",
        "Tell me about MBA programs",
        "What courses are available?",
        "Admission requirements for undergraduate programs"
    ]
    
    for query in test_queries:
        print(f"\\n❓ Query: {query}")
        
        # Find relevant chunks
        relevant_chunks = search_similar_chunks(query, processed_data)
        
        print(f"📚 Found {len(relevant_chunks)} relevant chunks:")
        
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"\\n  {i}. Source: {chunk['title']}")
            print(f"     Content: {chunk['content'][:200]}...")
            print(f"     Length: {chunk['content_length']} chars")
        
        # Simulate RAG response generation
        context = "\\n\\n".join([chunk['content'] for chunk in relevant_chunks])
        print(f"\\n🤖 This context would be sent to the AI model along with the query")
        print(f"📊 Total context length: {len(context)} characters")

def save_processed_data(processed_data: Dict[str, Any]):
    """Save processed data for future use."""
    os.makedirs('data/vectordb', exist_ok=True)
    
    # Save the processed chunks
    with open('data/processed/rag_chunks.json', 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": processed_data["timestamp"],
            "total_documents": processed_data["total_documents"],
            "chunks": processed_data["chunks"]
        }, f, indent=2, ensure_ascii=False)
    
    # Save vector index separately (in real implementation, this would be FAISS/ChromaDB)
    with open('data/vectordb/mock_vectors.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data["vector_index"], f, indent=2)
    
    print(f"\\n💾 Saved processed data:")
    print(f"  - data/processed/rag_chunks.json ({len(processed_data['chunks'])} chunks)")
    print(f"  - data/vectordb/mock_vectors.json (vector embeddings)")

def main():
    """Main demo function."""
    print("🎓 University AI Assistant - RAG Training Demonstration")
    print("=" * 60)
    
    print("\\nThis demo shows how the university data gets 'trained' using RAG:")
    print("1. ✅ Data Scraping: Already done (see data/raw/)")
    print("2. 🧩 Text Chunking: Break content into smaller pieces")
    print("3. 🔢 Embedding Creation: Convert text to numerical vectors")
    print("4. 💾 Vector Storage: Store in searchable database")
    print("5. 🔍 Retrieval: Find relevant chunks for queries")
    print("6. 🤖 Generation: AI uses chunks as context")
    
    # Load scraped data
    documents = load_scraped_data()
    
    if not documents:
        print("❌ No scraped data found. Run 'python test_scraper.py' first.")
        return
    
    # Process the data (this is the "training" part)
    processed_data = process_university_data(documents)
    
    print(f"\\n🎉 RAG 'Training' Complete!")
    print(f"📊 Total chunks created: {len(processed_data['chunks'])}")
    print(f"🔢 Vector dimensions: {len(next(iter(processed_data['vector_index'].values())))}")
    
    # Save processed data
    save_processed_data(processed_data)
    
    # Demo search functionality
    demo_rag_search(processed_data)
    
    print("\\n" + "=" * 60)
    print("🚀 RAG System Ready!")
    print("\\nIn the full system:")
    print("• Chunks would be stored in FAISS/ChromaDB for fast similarity search")
    print("• Real embeddings would be created using sentence-transformers")
    print("• AI models (Mistral/Groq) would generate responses using retrieved context")
    print("• New data can be added without retraining the base model")
    
    print("\\nNext steps:")
    print("1. Install full requirements: pip install -r requirements.txt")
    print("2. Run: python run.py")
    print("3. Ask questions about GD Goenka University!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
