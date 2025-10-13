#!/usr/bin/env python3
"""
Example showing LLM integration with file links
"""

from enhanced_rag_system import EnhancedRAGSystem

def simulate_llm_query(rag_system, user_query):
    """Simulate how an LLM would use the RAG system"""
    
    # Query the RAG system
    response = rag_system.query_with_files(user_query)
    
    # Build context for LLM
    context_prompt = f"""
User Query: {user_query}

Available Documents:
"""
    
    for ref in response['file_references']:
        context_prompt += f"""
- File: {ref['filename']}
- Type: {ref['content_type']} 
- Link: {ref['access_link']}
- Relevance: {ref['relevance_score']:.2f}
"""
    
    # Add RAG context
    if response['rag_results'].documents:
        context_prompt += f"\nRelevant Content Snippets:\n"
        for i, doc in enumerate(response['rag_results'].documents[:3]):
            context_prompt += f"{i+1}. {doc.content[:200]}...\n"
    
    return context_prompt

def main():
    print("🤖 LLM Integration Example")
    print("=" * 40)
    
    # Initialize RAG system
    rag = EnhancedRAGSystem("./llm_example_storage", port=8091)
    
    try:
        rag.start()
        
        # Example queries
        queries = [
            "What project information is available?",
            "Show me the system documentation",
            "What JSON files contain configuration data?"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 50)
            
            # Get LLM-ready context
            llm_context = simulate_llm_query(rag, query)
            print("📤 Context for LLM:")
            print(llm_context[:500] + "..." if len(llm_context) > 500 else llm_context)
    
    finally:
        rag.stop()

if __name__ == "__main__":
    main()
