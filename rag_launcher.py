#!/usr/bin/env python3
"""
RAG System Launcher
Lightweight launcher for the Enhanced RAG System
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_rag_system import EnhancedRAGSystem
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

try:
    from run_integrated_rag import IntegratedRAGSystem
    INTEGRATED_AVAILABLE = True
except ImportError:
    INTEGRATED_AVAILABLE = False


class RAGLauncher:
    """Simple launcher for RAG systems"""
    
    def __init__(self):
        self.current_system = None
        self.system_type = None
        
    def list_available_systems(self):
        """List available RAG systems"""
        print("🎯 Available RAG Systems:")
        
        if ENHANCED_AVAILABLE:
            print("  1. Enhanced RAG System (File Upload + Web Links)")
            print("     - File upload with web access")
            print("     - JSON/TXT processing")
            print("     - Direct file links for LLM")
        
        if INTEGRATED_AVAILABLE:
            print("  2. Integrated RAG System (Full Featured)")
            print("     - Complete RAG pipeline")
            print("     - Keystroke logging")
            print("     - OCR integration")
            print("     - Context-aware querying")
        
        if not ENHANCED_AVAILABLE and not INTEGRATED_AVAILABLE:
            print("  ❌ No RAG systems available")
            print("     Please ensure all required files are present")
    
    def launch_enhanced_rag(self, storage_path: str = "./rag_storage", port: int = 8089, test_mode: bool = False):
        """Launch the Enhanced RAG System"""
        if not ENHANCED_AVAILABLE:
            print("❌ Enhanced RAG System not available")
            return False
        
        print(f"🚀 Launching Enhanced RAG System...")
        print(f"   Storage: {storage_path}")
        print(f"   Web Port: {port}")
        
        try:
            self.current_system = EnhancedRAGSystem(Path(storage_path), port)
            self.system_type = "enhanced"
            
            self.current_system.start()
            
            if test_mode:
                self._run_enhanced_tests()
                return True
            
            print(f"\n✅ Enhanced RAG System running!")
            print(f"📁 Upload folder: {self.current_system.upload_folder}")
            print(f"🌐 Web interface: http://localhost:{port}")
            print(f"📋 File listing: http://localhost:{port}/files")
            print("\n💡 Usage:")
            print("   1. Place JSON files in uploads/json/")
            print("   2. Place TXT files in uploads/txt/")
            print("   3. Query the system to get file links")
            print("   4. LLM can access files via returned links")
            print("\nPress Ctrl+C to stop")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to launch Enhanced RAG System: {e}")
            return False
    
    def launch_integrated_rag(self, config_path: str = "rag_config.json", interface: str = "cli"):
        """Launch the Integrated RAG System"""
        if not INTEGRATED_AVAILABLE:
            print("❌ Integrated RAG System not available")
            return False
        
        print(f"🚀 Launching Integrated RAG System...")
        print(f"   Config: {config_path}")
        print(f"   Interface: {interface}")
        
        try:
            self.current_system = IntegratedRAGSystem(Path(config_path))
            self.system_type = "integrated"
            
            self.current_system.initialize_components()
            self.current_system.start_system()
            
            print(f"\n✅ Integrated RAG System running!")
            
            if interface == "gui":
                self.current_system.run_gui_interface()
            elif interface == "cli":
                self.current_system.run_cli_interface()
            else:
                print("🔄 Running in daemon mode...")
                try:
                    while self.current_system.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to launch Integrated RAG System: {e}")
            return False
    
    def stop_system(self):
        """Stop the current system"""
        if self.current_system:
            print("\n🛑 Stopping RAG system...")
            
            if self.system_type == "enhanced":
                self.current_system.stop()
            elif self.system_type == "integrated":
                self.current_system.stop_system()
            
            print("✅ System stopped")
            self.current_system = None
            self.system_type = None
        else:
            print("⚠️  No system running")
    
    def _run_enhanced_tests(self):
        """Run tests for enhanced RAG system"""
        print("\n🧪 Running Enhanced RAG Tests...")
        
        # Test file system
        files = self.current_system.file_manager.list_files()
        print(f"📁 Found {len(files)} uploaded files")
        
        for file_info in files:
            print(f"   📄 {file_info.filename} -> {self.current_system.file_manager.get_file_link(file_info.file_id)}")
        
        # Test query
        response = self.current_system.query_with_files("How do I use this system?")
        print(f"\n🔍 Test Query Results:")
        print(f"   RAG Documents: {len(response['rag_results'].documents)}")
        print(f"   File References: {len(response['file_references'])}")
        print(f"   Processing Time: {response['processing_time']:.3f}s")
        
        if response['file_references']:
            print("   📎 Available File Links:")
            for ref in response['file_references']:
                print(f"     • {ref['filename']} -> {ref['access_link']}")
        
        print("\n✅ Tests completed!")
    
    def interactive_menu(self):
        """Show interactive menu for system selection"""
        while True:
            print("\n" + "="*50)
            print("🎯 RAG System Launcher")
            print("="*50)
            
            self.list_available_systems()
            
            print("\n📋 Options:")
            if ENHANCED_AVAILABLE:
                print("  e) Launch Enhanced RAG (File Upload System)")
            if INTEGRATED_AVAILABLE:
                print("  i) Launch Integrated RAG (Full System)")
            print("  q) Quit")
            
            choice = input("\nSelect option: ").lower().strip()
            
            if choice == 'q':
                break
            elif choice == 'e' and ENHANCED_AVAILABLE:
                port = input("Web port [8089]: ").strip() or "8089"
                storage = input("Storage path [./rag_storage]: ").strip() or "./rag_storage"
                test = input("Run in test mode? [y/N]: ").lower().strip() == 'y'
                
                try:
                    if self.launch_enhanced_rag(storage, int(port), test):
                        if not test:
                            try:
                                while True:
                                    time.sleep(1)
                            except KeyboardInterrupt:
                                self.stop_system()
                except ValueError:
                    print("❌ Invalid port number")
                
            elif choice == 'i' and INTEGRATED_AVAILABLE:
                config = input("Config file [rag_config.json]: ").strip() or "rag_config.json"
                interface = input("Interface (gui/cli/daemon) [cli]: ").strip() or "cli"
                
                self.launch_integrated_rag(config, interface)
            
            else:
                print("❌ Invalid option or system not available")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG System Launcher")
    parser.add_argument("--system", choices=["enhanced", "integrated"], help="System to launch")
    parser.add_argument("--storage", default="./rag_storage", help="Storage path for Enhanced RAG")
    parser.add_argument("--port", type=int, default=8089, help="Web port for Enhanced RAG")
    parser.add_argument("--config", default="rag_config.json", help="Config file for Integrated RAG")
    parser.add_argument("--interface", choices=["gui", "cli", "daemon"], default="cli", help="Interface for Integrated RAG")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--interactive", action="store_true", help="Show interactive menu")
    
    args = parser.parse_args()
    
    launcher = RAGLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler():
        launcher.stop_system()
    
    try:
        if args.interactive:
            launcher.interactive_menu()
        elif args.system == "enhanced":
            if launcher.launch_enhanced_rag(args.storage, args.port, args.test):
                if not args.test:
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        launcher.stop_system()
        elif args.system == "integrated":
            launcher.launch_integrated_rag(args.config, args.interface)
        else:
            # Default: show menu
            launcher.interactive_menu()
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        launcher.stop_system()
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        launcher.stop_system()


if __name__ == "__main__":
    main()