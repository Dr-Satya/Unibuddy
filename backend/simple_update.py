#!/usr/bin/env python3
"""
Simple script to trigger knowledge base update
"""

import sys
import os
sys.path.append('src')

def main():
    print("🔄 Updating Knowledge Base...")
    
    try:
        # Import the university service
        from services import university_service
        
        # Trigger the update
        result = university_service.update_knowledge_base()
        
        if result.get('success'):
            print("✅ Knowledge base updated successfully!")
            print(f"📊 Scraped: {result.get('documents_scraped', 0)} documents")
            print(f"💾 Stored: {result.get('documents_stored', 0)} documents")
        else:
            print(f"❌ Update failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
