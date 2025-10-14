#!/usr/bin/env python3
"""
Integrated RAG System Launcher
Combines keystroke logging, OCR analysis, embedding generation, and intelligent querying
"""

import json
import time
import signal
import sys
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

# Import all our components
from ollama_rag_system import RAGSystem
from keystroke_logger import KeystrokeLogger
from embedding_pipeline import EmbeddingPipeline, KeystrokeEmbeddingProcessor
from rag_query_interface import RAGQueryInterface

# Import OCR system from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent / "Screenshare"))
try:
    from ocr_enhanced import OCRAssistant
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OCR system not available: {e}")
    OCR_AVAILABLE = False

try:
    from ollama_prompt_system import OllamaPromptSystem
    PROMPT_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Ollama prompt system not available: {e}")
    PROMPT_SYSTEM_AVAILABLE = False


class IntegratedRAGSystem:
    """Integrated RAG system combining all components"""
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger("IntegratedRAGSystem")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize storage paths
        self.storage_path = Path(self.config.get('storage_path', './rag_storage'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.rag_system = None
        self.embedding_pipeline = None
        self.keystroke_logger = None
        self.keystroke_processor = None
        self.query_interface = None
        self.ocr_assistant = None
        
        # System state
        self.running = False
        self.components_started = {}
        
        # Performance monitoring
        self.start_time = time.time()
        self.stats = {
            'keystrokes_processed': 0,
            'embeddings_generated': 0,
            'queries_processed': 0,
            'ocr_analyses': 0
        }
        
        print(f"🎯 Integrated RAG System initialized")
        print(f"   Storage: {self.storage_path}")
        print(f"   Config: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'storage_path': './rag_storage',
            'keystroke_logging': {
                'enabled': True,
                'privacy_mode': True,
                'session_timeout': 600,
                'batch_size': 5
            },
            'embedding_pipeline': {
                'chunk_size': 512,
                'cache_size': 1000,
                'batch_timeout': 30
            },
            'ocr_integration': {
                'enabled': True,
                'analysis_interval': 10,
                'confidence_threshold': 0.7
            },
            'query_interface': {
                'gui_enabled': True,
                'background_processing': True
            },
            'rag_system': {
                'vector_backend': 'auto',
                'max_documents': 10000,
                'similarity_threshold': 0.7
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge configs
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}, using defaults")
        else:
            # Save default config
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                self.logger.info(f"Created default configuration at {self.config_path}")
            except Exception as e:
                self.logger.warning(f"Could not save config file: {e}")
        
        return default_config
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            print("🚀 Initializing RAG System Components...")
            
            # 1. Initialize RAG System
            print("   📚 Initializing core RAG system...")
            rag_config = self.config.get('rag_system', {})
            self.rag_system = RAGSystem(
                self.storage_path / "rag_data",
                vector_backend=rag_config.get('vector_backend', 'auto'),
                max_documents=rag_config.get('max_documents', 10000)
            )
            self.components_started['rag_system'] = True
            print("   ✅ RAG system initialized")
            
            # 2. Initialize Embedding Pipeline  
            print("   🔢 Initializing embedding pipeline...")
            embedding_config = self.config.get('embedding_pipeline', {})
            self.embedding_pipeline = EmbeddingPipeline(
                self.storage_path,
                chunk_size=embedding_config.get('chunk_size', 512)
            )
            self.components_started['embedding_pipeline'] = True
            print("   ✅ Embedding pipeline initialized")
            
            # 3. Initialize Keystroke Logger (if enabled)
            keystroke_config = self.config.get('keystroke_logging', {})
            if keystroke_config.get('enabled', True):
                print("   ⌨️  Initializing keystroke logger...")
                
                # Create storage directory for keystroke data
                keystroke_storage = self.storage_path / "keystrokes"
                keystroke_storage.mkdir(exist_ok=True)
                
                self.keystroke_logger = KeystrokeLogger(
                    privacy_mode=keystroke_config.get('privacy_mode', True),
                    session_timeout_minutes=keystroke_config.get('session_timeout', 600)
                )
                
                # Initialize keystroke embedding processor
                self.keystroke_processor = KeystrokeEmbeddingProcessor(
                    self.embedding_pipeline,
                    self.rag_system
                )
                self.keystroke_processor.batch_size = keystroke_config.get('batch_size', 5)
                
                # Connect keystroke logger to processor
                self.keystroke_logger.set_rag_callback(self.keystroke_processor.add_keystroke_data)
                
                self.components_started['keystroke_logging'] = True
                print("   ✅ Keystroke logging initialized")
            
            # 4. Initialize OCR Assistant (if available and enabled)
            ocr_config = self.config.get('ocr_integration', {})
            if OCR_AVAILABLE and ocr_config.get('enabled', True):
                print("   👁️  Initializing OCR assistant...")
                try:
                    self.ocr_assistant = OCRAssistant()
                    self.components_started['ocr_integration'] = True
                    print("   ✅ OCR assistant initialized")
                except Exception as e:
                    self.logger.warning(f"OCR initialization failed: {e}")
            
            # 5. Initialize Query Interface
            print("   🧠 Initializing query interface...")
            self.query_interface = RAGQueryInterface(self.storage_path)
            self.components_started['query_interface'] = True
            print("   ✅ Query interface initialized")
            
            print(f"🎉 All components initialized successfully!")
            print(f"   Active components: {list(self.components_started.keys())}")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def start_system(self):
        """Start all system components"""
        if self.running:
            print("⚠️  System is already running")
            return
        
        try:
            print("▶️  Starting Integrated RAG System...")
            self.running = True
            
            # Start keystroke logging
            if self.keystroke_logger and 'keystroke_logging' in self.components_started:
                print("   Starting keystroke logging...")
                
                def start_keystroke_logging():
                    try:
                        self.keystroke_logger.start_logging()
                    except Exception as e:
                        self.logger.error(f"Keystroke logging error: {e}")
                
                keystroke_thread = threading.Thread(target=start_keystroke_logging, daemon=True)
                keystroke_thread.start()
                
                # Start keystroke processing
                self.keystroke_processor.start_background_processing()
                print("   ✅ Keystroke logging started")
            
            # Start OCR analysis (if enabled)
            if self.ocr_assistant and 'ocr_integration' in self.components_started:
                print("   Starting OCR analysis...")
                ocr_config = self.config.get('ocr_integration', {})
                
                def ocr_analysis_loop():
                    """Background OCR analysis loop"""
                    interval = ocr_config.get('analysis_interval', 10)
                    confidence_threshold = ocr_config.get('confidence_threshold', 0.7)
                    
                    while self.running:
                        try:
                            # Analyze current screen
                            analysis = self.ocr_assistant.analyze_screen()
                            
                            if analysis and analysis.get('confidence', 0) >= confidence_threshold:
                                # Process OCR text through embedding pipeline
                                ocr_text = analysis.get('text', '')
                                if ocr_text.strip():
                                    # Create metadata for OCR data
                                    metadata = {
                                        'confidence': analysis.get('confidence', 0),
                                        'activity_type': analysis.get('activity_type', 'unknown'),
                                        'window_title': analysis.get('window_title', ''),
                                        'timestamp': time.time()
                                    }
                                    
                                    # Generate embeddings and add to RAG system
                                    results = self.embedding_pipeline.process_content(
                                        ocr_text, 'ocr', metadata
                                    )
                                    
                                    if results:
                                        rag_documents = self.embedding_pipeline.create_rag_documents(results)
                                        self.rag_system.add_documents(rag_documents)
                                        self.stats['ocr_analyses'] += 1
                                        self.stats['embeddings_generated'] += len(results)
                            
                            time.sleep(interval)
                            
                        except Exception as e:
                            self.logger.error(f"OCR analysis error: {e}")
                            time.sleep(interval * 2)  # Wait longer on error
                
                ocr_thread = threading.Thread(target=ocr_analysis_loop, daemon=True)
                ocr_thread.start()
                print("   ✅ OCR analysis started")
            
            # Start background systems in query interface
            query_config = self.config.get('query_interface', {})
            if query_config.get('background_processing', True):
                print("   Starting background processing...")
                self.query_interface.start_background_systems()
                print("   ✅ Background processing started")
            
            print("🌟 Integrated RAG System is now running!")
            self._show_system_status()
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            self.stop_system()
            raise
    
    def stop_system(self):
        """Stop all system components gracefully"""
        if not self.running:
            return
        
        print("\n🛑 Stopping Integrated RAG System...")
        self.running = False
        
        try:
            # Stop keystroke logging
            if self.keystroke_logger:
                print("   Stopping keystroke logging...")
                self.keystroke_logger.stop_logging()
            
            if self.keystroke_processor:
                print("   Stopping keystroke processing...")
                self.keystroke_processor.stop_background_processing()
            
            # Stop query interface
            if self.query_interface:
                print("   Stopping query interface...")
                self.query_interface.shutdown()
            
            # Shutdown RAG system
            if self.rag_system:
                print("   Stopping RAG system...")
                self.rag_system.shutdown()
            
            print("✅ System stopped gracefully")
            self._show_final_stats()
            
        except Exception as e:
            self.logger.error(f"System shutdown error: {e}")
    
    def run_gui_interface(self):
        """Run the GUI interface"""
        if not self.query_interface:
            print("❌ Query interface not initialized")
            return
        
        query_config = self.config.get('query_interface', {})
        if not query_config.get('gui_enabled', True):
            print("⚠️  GUI interface disabled in configuration")
            return
        
        print("🖥️  Launching GUI interface...")
        try:
            self.query_interface.run_gui()
        except KeyboardInterrupt:
            print("\n👋 GUI interface closed by user")
        except Exception as e:
            self.logger.error(f"GUI interface error: {e}")
    
    def run_cli_interface(self):
        """Run the CLI interface"""
        print("💻 CLI Interface - Type 'help' for commands, 'quit' to exit")
        
        while self.running:
            try:
                user_input = input("\n🧠 RAG> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_cli_help()
                elif user_input.lower() == 'status':
                    self._show_system_status()
                elif user_input.lower() == 'stats':
                    self._show_detailed_stats()
                elif user_input.startswith('query '):
                    query_text = user_input[6:].strip()
                    if query_text:
                        self._process_cli_query(query_text)
                else:
                    # Treat as query
                    self._process_cli_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 CLI interface interrupted")
                break
            except EOFError:
                print("\n👋 CLI interface closed")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _process_cli_query(self, query_text: str):
        """Process a query in CLI mode"""
        try:
            print(f"🔍 Processing query: {query_text}")
            
            # Build context and process query
            context = self.query_interface.context_aggregator.build_query_context(query_text)
            response = self.query_interface.query_processor.process_query(query_text, context)
            
            # Display response
            print(f"\n💡 Response:")
            print(response.combined_insight)
            
            if response.follow_up_queries:
                print(f"\n🔄 Follow-up suggestions:")
                for i, followup in enumerate(response.follow_up_queries, 1):
                    print(f"   {i}. {followup}")
            
            print(f"\n📊 Confidence: {response.confidence_score:.1%} | Time: {response.processing_time:.2f}s")
            
            self.stats['queries_processed'] += 1
            
        except Exception as e:
            print(f"❌ Query processing error: {e}")
    
    def _show_cli_help(self):
        """Show CLI help"""
        print("""
📖 CLI Commands:
   help        - Show this help message
   status      - Show system status
   stats       - Show detailed statistics
   query <Q>   - Process a specific query
   quit/exit   - Stop the system and exit
   
💡 You can also type any text directly as a query.
        """)
    
    def _show_system_status(self):
        """Show current system status"""
        uptime = time.time() - self.start_time
        uptime_str = f"{uptime/3600:.1f} hours" if uptime > 3600 else f"{uptime/60:.1f} minutes"
        
        print(f"\n📊 System Status:")
        print(f"   Uptime: {uptime_str}")
        print(f"   Running: {self.running}")
        print(f"   Active Components: {len(self.components_started)}")
        
        for component, active in self.components_started.items():
            status_icon = "✅" if active else "❌"
            print(f"     {status_icon} {component.replace('_', ' ').title()}")
        
        print(f"\n📈 Quick Stats:")
        print(f"   Queries Processed: {self.stats['queries_processed']}")
        print(f"   Embeddings Generated: {self.stats['embeddings_generated']}")
        print(f"   OCR Analyses: {self.stats['ocr_analyses']}")
    
    def _show_detailed_stats(self):
        """Show detailed system statistics"""
        self._show_system_status()
        
        # RAG system stats
        if self.rag_system:
            rag_stats = self.rag_system.get_system_stats()
            print(f"\n🗄️  RAG System Details:")
            print(f"   Total Documents: {rag_stats['vector_store']['total_documents']}")
            print(f"   Backend: {rag_stats['vector_store']['backend']}")
            print(f"   Total Queries: {rag_stats['query_stats']['total_queries']}")
            print(f"   Avg Retrieval Time: {rag_stats['query_stats']['avg_retrieval_time']:.3f}s")
        
        # Embedding pipeline stats
        if self.embedding_pipeline:
            embedding_stats = self.embedding_pipeline.get_stats()
            print(f"\n🔢 Embedding Pipeline Details:")
            print(f"   Total Processed: {embedding_stats['total_processed']}")
            print(f"   Cache Size: {embedding_stats['cache_size']}")
            print(f"   Cache Hit Ratio: {embedding_stats['cache_hit_ratio']:.1%}")
            print(f"   Available Embedders: {embedding_stats['available_embedders']}")
            print(f"   Model Usage: {embedding_stats['model_usage']}")
        
        # Keystroke logger stats
        if self.keystroke_logger and hasattr(self.keystroke_logger, 'get_session_stats'):
            keystroke_stats = self.keystroke_logger.get_session_stats()
            print(f"\n⌨️  Keystroke Logger Details:")
            for key, value in keystroke_stats.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
    
    def _show_final_stats(self):
        """Show final statistics on shutdown"""
        uptime = time.time() - self.start_time
        print(f"\n📊 Final Statistics:")
        print(f"   Total Uptime: {uptime/3600:.2f} hours")
        print(f"   Queries Processed: {self.stats['queries_processed']}")
        print(f"   Embeddings Generated: {self.stats['embeddings_generated']}")
        print(f"   OCR Analyses: {self.stats['ocr_analyses']}")
        
        if self.stats['queries_processed'] > 0:
            print(f"   Avg Queries/Hour: {self.stats['queries_processed'] / (uptime/3600):.1f}")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    if 'integrated_system' in globals():
        integrated_system.stop_system()
    sys.exit(0)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Integrated RAG System")
    parser.add_argument("--config", type=Path, default="rag_config.json", help="Configuration file path")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--cli", action="store_true", help="Launch CLI interface")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--init-only", action="store_true", help="Initialize components only, don't start")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create integrated system
        global integrated_system
        integrated_system = IntegratedRAGSystem(args.config)
        
        # Initialize components
        integrated_system.initialize_components()
        
        if args.init_only:
            print("✅ Components initialized successfully. Exiting.")
            return
        
        # Start the system
        integrated_system.start_system()
        
        # Choose interface mode
        if args.gui:
            integrated_system.run_gui_interface()
        elif args.cli:
            integrated_system.run_cli_interface()
        elif args.daemon:
            print("🔄 Running in daemon mode. Press Ctrl+C to stop.")
            try:
                while integrated_system.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            # Default: show menu
            print("\n🎯 Integrated RAG System Menu:")
            print("   1. Launch GUI Interface")
            print("   2. Launch CLI Interface") 
            print("   3. Run as background daemon")
            print("   4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                integrated_system.run_gui_interface()
            elif choice == '2':
                integrated_system.run_cli_interface()
            elif choice == '3':
                print("🔄 Running in daemon mode. Press Ctrl+C to stop.")
                try:
                    while integrated_system.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            else:
                print("👋 Exiting...")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        logging.exception("System error")
    finally:
        # Ensure cleanup
        if 'integrated_system' in locals():
            integrated_system.stop_system()


if __name__ == "__main__":
    main()