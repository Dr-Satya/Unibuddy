#!/usr/bin/env python3
"""
Debug Dr. Ugur Guven retrieval to ensure proper responses
"""

import sys
sys.path.append('src')
from main import rag_system

def debug_ugur_retrieval():
    print("🔍 DEBUGGING DR. UGUR GUVEN RETRIEVAL")
    print("=" * 50)
    
    # Test specific query
    query = "Dr. Ugur Guven"
    print(f"Query: {query}")
    print("-" * 30)
    
    # Get search results
    results = rag_system.search(query, top_k=5)
    
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result.score:.3f}) ---")
        print(f"Source: {result.source}")
        print(f"Content: {result.content[:500]}...")
        
        if result.extra_metadata:
            print(f"Title: {result.extra_metadata.get('title', 'N/A')}")
    
    # Test context generation
    print("\n" + "=" * 50)
    print("CONTEXT GENERATION TEST")
    print("=" * 50)
    
    context = rag_system.get_context_for_query(query)
    print(f"Generated context ({len(context)} chars):")
    print(context)
    
    # Check for specific content
    print("\n" + "=" * 50)
    print("CONTENT ANALYSIS")
    print("=" * 50)
    
    context_lower = context.lower()
    
    # More comprehensive faculty indicators
    faculty_keywords = [
        'ugur', 'guven', 'aerospace', 'engineering', 'professor', 
        'dr.', 'dr ', 'expertise', 'research', 'space', 'satellite',
        'propulsion', 'technology', 'faculty', 'academic', 'university'
    ]
    
    found_keywords = []
    for keyword in faculty_keywords:
        if keyword in context_lower:
            found_keywords.append(keyword)
    
    print(f"Keywords found in context: {found_keywords}")
    
    # Check if we have enough indicators for Dr. Ugur Guven
    ugur_specific = ['ugur', 'guven', 'aerospace']
    ugur_found = [kw for kw in ugur_specific if kw in found_keywords]
    
    print(f"Dr. Ugur Guven specific indicators: {ugur_found}")
    
    if len(ugur_found) >= 2:
        print("✅ Dr. Ugur Guven data is properly retrievable!")
    else:
        print("⚠️ Need to check Dr. Ugur Guven data in knowledge base")

if __name__ == "__main__":
    debug_ugur_retrieval()
