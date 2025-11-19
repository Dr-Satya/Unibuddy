#!/usr/bin/env python3
"""
University Knowledge Base Setup Script

This script scrapes data from GD Goenka University website and sets up the RAG knowledge base.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def setup_environment():
    """Create necessary directories and setup environment."""
    print("🔧 Setting up environment...")
    
    # Create directories
    directories = [
        'data/raw',
        'data/processed', 
        'data/vectordb',
        'logs',
        'secure_storage'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def scrape_university_data():
    """Scrape university data from the website."""
    print("\\n🕷️ Starting university data scraping...")
    
    try:
        from scraper import university_scraper
        from database import db_manager
        
        # Scrape all university data
        documents = university_scraper.scrape_all()
        
        if not documents:
            print("❌ No documents were scraped!")
            return False
        
        print(f"✅ Successfully scraped {len(documents)} documents")
        
        # Display scraped documents info
        for doc in documents:
            print(f"  📄 {doc['title']} ({len(doc['content'])} chars)")
        
        # Store in database
        stored_count = university_scraper.store_scraped_data(documents)
        print(f"💾 Stored {stored_count} documents in database")
        
        return len(documents) > 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install requirements: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return False

def setup_rag_system():
    """Initialize and populate the RAG system."""
    print("\\n🧠 Setting up RAG system...")
    
    try:
        from rag import rag_system
        from database import db_manager
        
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag_system._initialize()
        
        if not rag_system.is_initialized:
            print("❌ Failed to initialize RAG system")
            return False
        
        # Update RAG system with scraped data
        print("📊 Processing university data for RAG...")
        rag_system.update_from_database()
        
        # Get statistics
        stats = rag_system.get_statistics()
        print(f"✅ RAG system ready:")
        print(f"  📊 Total vectors: {stats['total_vectors']}")
        print(f"  📚 Total documents: {stats['total_documents']}")
        print(f"  🔤 Embedding model: {stats['embedding_model']}")
        
        return stats['total_vectors'] > 0
        
    except Exception as e:
        print(f"❌ Error setting up RAG system: {e}")
        return False

def test_knowledge_base():
    """Test the knowledge base with sample queries."""
    print("\\n🧪 Testing knowledge base...")
    
    try:
        from services import chat_service
        
        test_queries = [
            "What is the fee structure at GD Goenka University?",
            "Tell me about admission requirements",
            "What courses are available?",
        ]
        
        for query in test_queries:
            print(f"\\n❓ Query: {query}")
            
            try:
                response = chat_service.chat(query, user_id="test_user")
                print(f"🤖 Response: {response.message[:200]}...")
                print(f"📊 Model: {response.model_used}, Tokens: {response.tokens_used}")
                
                if response.sources:
                    print(f"📚 Sources: {len(response.sources)} found")
                
            except Exception as e:
                print(f"❌ Error with query: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing knowledge base: {e}")
        return False

def save_scraped_data_summary():
    """Save a summary of scraped data to data folder."""
    try:
        from database import db_manager
        
        print("\\n💾 Saving scraped data summary...")
        
        university_data = db_manager.get_university_data()
        
        summary = {
            "scrape_timestamp": datetime.now().isoformat(),
            "total_documents": len(university_data),
            "documents": []
        }
        
        for data in university_data:
            doc_info = {
                "title": data.title,
                "url": data.url,
                "content_type": data.content_type,
                "content_length": len(data.content) if data.content else 0,
                "scraped_at": data.scraped_at.isoformat() if data.scraped_at else None
            }
            summary["documents"].append(doc_info)
        
        # Save to data/processed folder
        import json
        with open('data/processed/scraped_data_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("✅ Saved data summary to data/processed/scraped_data_summary.json")
        
        # Also save full content for reference
        for data in university_data:
            filename = f"data/raw/{data.content_type}_{data.id[:8]}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {data.title}\\n")
                f.write(f"URL: {data.url}\\n")
                f.write(f"Scraped: {data.scraped_at}\\n")
                f.write("-" * 50 + "\\n")
                f.write(data.content)
            
            print(f"💾 Saved raw content to {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving data summary: {e}")
        return False

def main():
    """Main setup function."""
    print("🎓 University Assistant - Knowledge Base Setup")
    print("=" * 50)
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Scrape university data
    if not scrape_university_data():
        print("❌ Failed to scrape university data. Exiting.")
        return False
    
    # Step 3: Save scraped data to files
    save_scraped_data_summary()
    
    # Step 4: Setup RAG system
    if not setup_rag_system():
        print("❌ Failed to setup RAG system. Exiting.")
        return False
    
    # Step 5: Test the knowledge base
    test_knowledge_base()
    
    print("\\n🎉 Knowledge Base Setup Complete!")
    print("\\nNext steps:")
    print("1. Run 'python run.py' to start the chatbot")
    print("2. Ask questions about GD Goenka University")
    print("3. The system will use RAG to provide context-aware answers")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
