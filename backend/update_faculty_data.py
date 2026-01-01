#!/usr/bin/env python3
"""
Script to update the knowledge base with faculty data
"""

import sys
import os
sys.path.append('src')

from scraper import university_scraper
from main import RAGSystem

def main():
    print("🔄 Updating Knowledge Base with Faculty Data")
    print("=" * 50)
    
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    # Scrape faculty data specifically
    print("📡 Scraping faculty data...")
    faculty_doc = university_scraper.scrape_faculty_data()
    
    if faculty_doc:
        print(f"✅ Successfully scraped faculty data: {len(faculty_doc['content'])} characters")
        
        # Store the faculty document
        print("💾 Storing faculty data in database...")
        university_scraper.store_scraped_data([faculty_doc])
        
        # Update RAG system with new data
        print("🧠 Updating RAG system...")
        try:
            rag_system.update_knowledge_base([faculty_doc])
            print("✅ Knowledge base updated successfully!")
            
            # Test the updated system
            print("\n🧪 Testing the updated system...")
            test_queries = [
                "who is dr ugur",
                "tell me about faculty members",
                "who are the engineering professors"
            ]
            
            for query in test_queries:
                print(f"\n🔍 Testing query: '{query}'")
                try:
                    context = rag_system.get_relevant_context(query)
                    if context:
                        print(f"   ✅ Found {len(context)} relevant sources")
                        # Show a preview of the first source
                        preview = context[0]['content'][:200] + "..." if len(context[0]['content']) > 200 else context[0]['content']
                        print(f"   📄 Preview: {preview}")
                        
                        # Generate response
                        response = rag_system.generate_response(query)
                        response_preview = response[:300] + "..." if len(response) > 300 else response
                        print(f"   🤖 Response: {response_preview}")
                    else:
                        print("   ❌ No relevant sources found")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
        except Exception as e:
            print(f"❌ Error updating RAG system: {e}")
    else:
        print("❌ Failed to scrape faculty data")

if __name__ == "__main__":
    main()
