#!/usr/bin/env python3
"""
Test faculty queries to verify Dr. Ugur Guven and other faculty data integration
"""

import sys
sys.path.append('src')
from main import rag_system

def main():
    print('🤖 Testing Faculty Query System')
    print('=' * 50)
    
    # Use the global RAG system instance
    
    # Test queries specifically for faculty
    queries = [
        'Who is Dr. Ugur Guven?',
        'Tell me about aerospace professors',
        'What is Dr. Ugur Guven expertise?',
        'Who are the faculty members in aerospace engineering?',
        'Tell me about Dr. Ugur Guven background',
        'What programs does Dr. Ugur Guven teach?'
    ]
    
    for query in queries:
        print(f'\n📝 Query: {query}')
        print('-' * 40)
        try:
            # Get relevant documents using the search method
            results = rag_system.search(query)
            
            if results and len(results) > 0:
                print(f'✅ Found {len(results)} relevant documents')
                
                # Show first source preview
                first_result = results[0]
                preview = first_result.content[:200] + "..." if len(first_result.content) > 200 else first_result.content
                print(f'Sample content: {preview}')
                print(f'Similarity score: {first_result.score:.3f}')
                
                # Get formatted context
                context = rag_system.get_context_for_query(query)
                print(f'Formatted context length: {len(context)} characters')
                
            else:
                print('❌ No relevant documents found')
                
        except Exception as e:
            print(f'❌ Error: {e}')
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    main()
