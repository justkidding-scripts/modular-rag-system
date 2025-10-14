#!/usr/bin/env python3
"""
Basic usage example for Modular RAG System
"""

from enhanced_rag_system import EnhancedRAGSystem
import time

def main():
    print("🎯 Modular RAG System - Basic Usage Example")
    print("=" * 50)
    
    # Initialize system
    print("\n📚 Initializing RAG system...")
    rag = EnhancedRAGSystem("./example_storage", port=8090)
    
    try:
        # Start system
        rag.start()
        print("✅ System started successfully")
        
        # Query the system
        print("\n🔍 Querying system...")
        response = rag.query_with_files("What files are available?")
        
        print(f"📊 Query Results:")
        print(f"   - RAG Documents: {len(response['rag_results'].documents)}")
        print(f"   - File References: {len(response['file_references'])}")
        print(f"   - Processing Time: {response['processing_time']:.3f}s")
        
        if response['file_references']:
            print("\n📎 Available File Links:")
            for ref in response['file_references']:
                print(f"   • {ref['filename']} -> {ref['access_link']}")
        
        # Show system stats
        stats = rag.get_system_stats()
        print(f"\n📈 System Statistics:")
        print(f"   - Total Files: {stats['file_system']['total_files']}")
        print(f"   - Web Server: {'Running' if stats['file_system']['web_server_running'] else 'Stopped'}")
        print(f"   - Total Embeddings: {stats['file_system']['total_embeddings']}")
        
        print("\n✅ Example completed successfully!")
        print("🌐 Web interface available at: http://localhost:8090")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        rag.stop()
        print("🛑 System stopped")

if __name__ == "__main__":
    main()
