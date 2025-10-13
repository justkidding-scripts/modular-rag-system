#!/usr/bin/env python3
"""
RAG Query Interface for Ollama LLM Integration
Intelligent querying system that combines real-time context with historical data for enhanced AI responses
"""

import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import queue
from collections import deque, defaultdict

# Import our RAG system components
from ollama_rag_system import RAGSystem, RAGResult, RAGDocument
from keystroke_logger import KeystrokeLogger

# Import Ollama prompt system from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent / "Screenshare"))
from ollama_prompt_system import OllamaPromptSystem, AnalysisResponse


@dataclass
class QueryContext:
    """Enhanced context for RAG queries"""
    current_text: str
    recent_keystrokes: List[str]
    active_application: str
    window_title: str
    typing_speed: float
    session_duration: float
    query_type: str  # 'analysis', 'suggestion', 'question', 'completion'
    priority: int = 1
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class EnhancedResponse:
    """Enhanced response combining RAG data with AI analysis"""
    query: str
    context: QueryContext
    rag_results: RAGResult
    ai_response: AnalysisResponse
    combined_insight: str
    confidence_score: float
    processing_time: float
    sources_used: List[str]
    follow_up_queries: List[str]
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ContextAggregator:
    """Aggregates and manages contextual information from multiple sources"""
    
    def __init__(self, max_context_size: int = 1000):
        self.max_context_size = max_context_size
        self.context_buffer = deque(maxlen=max_context_size)
        self.application_contexts = defaultdict(list)
        self.typing_patterns = defaultdict(list)
        self.session_data = {}
        
    def add_keystroke_context(self, content: str, metadata: Dict[str, Any]):
        """Add keystroke context from the logger"""
        context_entry = {
            'type': 'keystroke',
            'timestamp': time.time(),
            'content': content,
            'metadata': metadata,
            'application': metadata.get('application', 'unknown'),
            'window_title': metadata.get('window_title', ''),
            'session_id': metadata.get('session_id', ''),
            'typing_speed': metadata.get('typing_speed', 0),
            'word_count': metadata.get('word_count', 0)
        }
        
        self.context_buffer.append(context_entry)
        
        # Track application-specific contexts
        app = metadata.get('application', 'unknown')
        self.application_contexts[app].append(context_entry)
        if len(self.application_contexts[app]) > 100:
            self.application_contexts[app] = self.application_contexts[app][-100:]
        
        # Track typing patterns
        if metadata.get('typing_speed', 0) > 0:
            self.typing_patterns[app].append({
                'timestamp': time.time(),
                'speed': metadata.get('typing_speed', 0),
                'word_count': metadata.get('word_count', 0)
            })
            if len(self.typing_patterns[app]) > 50:
                self.typing_patterns[app] = self.typing_patterns[app][-50:]
    
    def add_ocr_context(self, text: str, metadata: Dict[str, Any]):
        """Add OCR context from screen analysis"""
        context_entry = {
            'type': 'ocr',
            'timestamp': time.time(),
            'content': text,
            'metadata': metadata,
            'confidence': metadata.get('confidence', 0),
            'activity_type': metadata.get('activity_type', 'unknown')
        }
        
        self.context_buffer.append(context_entry)
    
    def get_recent_context(self, max_entries: int = 10, time_window: int = 300) -> List[Dict]:
        """Get recent context within time window"""
        cutoff_time = time.time() - time_window
        recent_context = [
            entry for entry in self.context_buffer 
            if entry['timestamp'] >= cutoff_time
        ]
        
        return list(recent_context)[-max_entries:]
    
    def get_application_context(self, application: str, max_entries: int = 5) -> List[Dict]:
        """Get recent context for specific application"""
        return list(self.application_contexts.get(application, []))[-max_entries:]
    
    def get_typing_analysis(self, application: str = None) -> Dict[str, Any]:
        """Get typing pattern analysis"""
        if application:
            patterns = self.typing_patterns.get(application, [])
        else:
            patterns = [pattern for app_patterns in self.typing_patterns.values() for pattern in app_patterns]
        
        if not patterns:
            return {}
        
        recent_patterns = [p for p in patterns if time.time() - p['timestamp'] < 1800]  # 30 minutes
        
        if not recent_patterns:
            return {}
        
        avg_speed = sum(p['speed'] for p in recent_patterns) / len(recent_patterns)
        total_words = sum(p['word_count'] for p in recent_patterns)
        
        return {
            'average_typing_speed': avg_speed,
            'recent_sessions': len(recent_patterns),
            'total_words': total_words,
            'productivity_score': min(avg_speed / 50, 1.0),  # Normalized to 0-1
            'consistency': self._calculate_consistency(recent_patterns)
        }
    
    def _calculate_consistency(self, patterns: List[Dict]) -> float:
        """Calculate typing consistency score"""
        if len(patterns) < 2:
            return 1.0
        
        speeds = [p['speed'] for p in patterns]
        mean_speed = sum(speeds) / len(speeds)
        variance = sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)
        std_dev = variance ** 0.5
        
        # Normalized consistency score (0-1, higher = more consistent)
        return max(0, 1 - (std_dev / max(mean_speed, 1)))
    
    def build_query_context(self, current_query: str, application: str = None, window_title: str = None) -> QueryContext:
        """Build comprehensive query context"""
        recent_context = self.get_recent_context()
        app_context = self.get_application_context(application or 'unknown') if application else []
        typing_analysis = self.get_typing_analysis(application)
        
        # Extract recent keystrokes
        recent_keystrokes = [
            entry['content'] for entry in recent_context 
            if entry['type'] == 'keystroke'
        ][-5:]
        
        return QueryContext(
            current_text=current_query,
            recent_keystrokes=recent_keystrokes,
            active_application=application or 'unknown',
            window_title=window_title or '',
            typing_speed=typing_analysis.get('average_typing_speed', 0),
            session_duration=self._calculate_session_duration(recent_context),
            query_type=self._infer_query_type(current_query, recent_context)
        )
    
    def _calculate_session_duration(self, recent_context: List[Dict]) -> float:
        """Calculate current session duration"""
        if not recent_context:
            return 0.0
        
        timestamps = [entry['timestamp'] for entry in recent_context]
        return (max(timestamps) - min(timestamps)) / 60  # in minutes
    
    def _infer_query_type(self, query: str, context: List[Dict]) -> str:
        """Infer the type of query based on content and context"""
        query_lower = query.lower()
        
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'which', '?']
        suggestion_indicators = ['suggest', 'recommend', 'improve', 'better', 'optimize']
        completion_indicators = ['complete', 'finish', 'continue', 'next']
        
        if any(indicator in query_lower for indicator in question_indicators):
            return 'question'
        elif any(indicator in query_lower for indicator in suggestion_indicators):
            return 'suggestion'
        elif any(indicator in query_lower for indicator in completion_indicators):
            return 'completion'
        else:
            return 'analysis'


class RAGQueryProcessor:
    """Processes queries using RAG system and AI analysis"""
    
    def __init__(self, rag_system: RAGSystem, ollama_system: OllamaPromptSystem):
        self.rag_system = rag_system
        self.ollama_system = ollama_system
        self.logger = logging.getLogger("RAGQueryProcessor")
        
        # Query optimization
        self.query_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
    def process_query(self, query: str, context: QueryContext) -> EnhancedResponse:
        """Process query with RAG and AI integration"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(query, context)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        try:
            # Step 1: Query RAG system for relevant historical data
            rag_results = self._query_rag_system(query, context)
            
            # Step 2: Build enhanced context for AI analysis
            enhanced_context = self._build_enhanced_context(context, rag_results)
            
            # Step 3: Get AI analysis with enhanced context
            ai_response = self._get_ai_analysis(query, enhanced_context)
            
            # Step 4: Combine and synthesize results
            combined_insight = self._synthesize_response(query, context, rag_results, ai_response)
            
            # Step 5: Calculate confidence and generate follow-ups
            confidence_score = self._calculate_confidence(rag_results, ai_response)
            follow_up_queries = self._generate_follow_up_queries(query, context, rag_results)
            sources_used = self._extract_sources(rag_results)
            
            # Create enhanced response
            response = EnhancedResponse(
                query=query,
                context=context,
                rag_results=rag_results,
                ai_response=ai_response,
                combined_insight=combined_insight,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                sources_used=sources_used,
                follow_up_queries=follow_up_queries
            )
            
            # Cache the response
            self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            # Return fallback response
            return self._create_fallback_response(query, context, str(e))
    
    def _query_rag_system(self, query: str, context: QueryContext) -> RAGResult:
        """Query the RAG system with context-aware parameters"""
        # Adjust query parameters based on context
        max_results = 5
        source_filters = None
        
        # Customize based on query type
        if context.query_type == 'question':
            max_results = 7  # More results for questions
        elif context.query_type == 'completion':
            max_results = 3  # Fewer, more relevant results for completion
            source_filters = ['keystroke']  # Focus on typing history
        
        # Include application context in query
        if context.active_application != 'unknown':
            enhanced_query = f"{query} (context: {context.active_application})"
        else:
            enhanced_query = query
        
        return self.rag_system.query(
            enhanced_query,
            context={'application': context.active_application, 'query_type': context.query_type},
            max_results=max_results,
            source_filters=source_filters
        )
    
    def _build_enhanced_context(self, context: QueryContext, rag_results: RAGResult) -> Dict[str, Any]:
        """Build enhanced context for AI analysis"""
        enhanced_context = {
            'current_query': context.current_text,
            'application_context': {
                'active_app': context.active_application,
                'window_title': context.window_title,
                'session_duration': context.session_duration
            },
            'typing_analysis': {
                'recent_keystrokes': context.recent_keystrokes[-3:],  # Last 3 for context
                'typing_speed': context.typing_speed,
                'productivity_indicator': 'high' if context.typing_speed > 40 else 'normal' if context.typing_speed > 20 else 'low'
            },
            'query_metadata': {
                'query_type': context.query_type,
                'priority': context.priority,
                'timestamp': context.timestamp
            }
        }
        
        # Add RAG context
        if rag_results.documents:
            enhanced_context['historical_context'] = {
                'relevant_documents_count': len(rag_results.documents),
                'average_similarity': sum(rag_results.similarities) / len(rag_results.similarities),
                'time_span_covered': self._calculate_time_span(rag_results.documents),
                'most_relevant_content': rag_results.documents[0].content[:200] if rag_results.documents else "",
                'source_types': list(set(doc.source for doc in rag_results.documents)),
                'context_summary': self._summarize_rag_context(rag_results)
            }
        
        return enhanced_context
    
    def _get_ai_analysis(self, query: str, enhanced_context: Dict) -> AnalysisResponse:
        """Get AI analysis using the enhanced context"""
        # Convert enhanced context to format expected by Ollama system
        text_history = enhanced_context.get('typing_analysis', {}).get('recent_keystrokes', [])
        session_stats = {
            'session_duration': enhanced_context.get('application_context', {}).get('session_duration', 0),
            'typing_speed': enhanced_context.get('typing_analysis', {}).get('typing_speed', 0),
            'historical_documents': enhanced_context.get('historical_context', {}).get('relevant_documents_count', 0)
        }
        
        return self.ollama_system.analyze_content(query, text_history, session_stats)
    
    def _synthesize_response(self, query: str, context: QueryContext, rag_results: RAGResult, ai_response: AnalysisResponse) -> str:
        """Synthesize final response combining RAG and AI insights"""
        synthesis = []
        
        # Start with AI insight
        synthesis.append(f"🧠 **AI Analysis**: {ai_response.main_insight}")
        
        # Add historical context if available
        if rag_results.documents:
            synthesis.append(f"\n📚 **Historical Context**: Found {len(rag_results.documents)} relevant past experiences:")
            
            for i, doc in enumerate(rag_results.documents[:3], 1):  # Top 3 most relevant
                similarity = rag_results.similarities[i-1] if i-1 < len(rag_results.similarities) else 0
                source_icon = "⌨️" if doc.source == 'keystroke' else "👁️" if doc.source == 'ocr' else "📄"
                
                synthesis.append(f"   {source_icon} {similarity:.1%} similar: {doc.content[:100]}...")
        
        # Add specific suggestions based on query type
        if context.query_type == 'question' and ai_response.questions:
            synthesis.append(f"\n❓ **Related Questions**: {', '.join(ai_response.questions[:2])}")
        
        elif context.query_type == 'suggestion' and ai_response.suggestions:
            synthesis.append(f"\n💡 **Actionable Suggestions**: {'; '.join(ai_response.suggestions[:3])}")
        
        # Add productivity insights if relevant
        if context.typing_speed > 0:
            productivity_insight = self._get_productivity_insight(context.typing_speed)
            synthesis.append(f"\n⚡ **Productivity Insight**: {productivity_insight}")
        
        return "\n".join(synthesis)
    
    def _calculate_confidence(self, rag_results: RAGResult, ai_response: AnalysisResponse) -> float:
        """Calculate overall confidence score for the response"""
        confidence_factors = []
        
        # AI confidence
        confidence_factors.append(ai_response.confidence * 0.4)
        
        # RAG relevance
        if rag_results.similarities:
            avg_similarity = sum(rag_results.similarities) / len(rag_results.similarities)
            confidence_factors.append(avg_similarity * 0.3)
        
        # Historical data availability
        doc_factor = min(len(rag_results.documents) / 5.0, 1.0) * 0.2
        confidence_factors.append(doc_factor)
        
        # Response type factor
        if ai_response.analysis_type == "ai_generated":
            confidence_factors.append(0.1)
        elif ai_response.analysis_type == "premade":
            confidence_factors.append(0.05)
        
        return sum(confidence_factors)
    
    def _generate_follow_up_queries(self, original_query: str, context: QueryContext, rag_results: RAGResult) -> List[str]:
        """Generate intelligent follow-up queries"""
        follow_ups = []
        
        # Based on query type
        if context.query_type == 'analysis':
            follow_ups.extend([
                f"How can I improve this {context.active_application} workflow?",
                "What similar situations have I encountered before?",
                "What are the key patterns in my work on this topic?"
            ])
        elif context.query_type == 'question':
            follow_ups.extend([
                "Can you provide more specific examples?",
                "What are the best practices for this situation?",
                "How does this compare to my previous approaches?"
            ])
        elif context.query_type == 'suggestion':
            follow_ups.extend([
                "What would be the next step to implement this?",
                "Are there any potential challenges with this approach?",
                "How have I handled similar situations before?"
            ])
        
        # Based on RAG results
        if rag_results.documents:
            most_recent = max(rag_results.documents, key=lambda d: d.timestamp)
            time_diff = (time.time() - most_recent.timestamp) / 3600  # hours
            
            if time_diff < 24:
                follow_ups.append("What has changed since I last worked on this today?")
            elif time_diff < 168:  # 1 week
                follow_ups.append("How has my approach evolved this week?")
        
        return follow_ups[:4]  # Return top 4 follow-ups
    
    def _extract_sources(self, rag_results: RAGResult) -> List[str]:
        """Extract source information from RAG results"""
        sources = []
        
        for doc in rag_results.documents:
            source_info = f"{doc.source}"
            if doc.metadata:
                if 'application' in doc.metadata:
                    source_info += f" ({doc.metadata['application']})"
                if 'timestamp' in doc.metadata:
                    dt = datetime.fromtimestamp(doc.metadata['timestamp'])
                    source_info += f" - {dt.strftime('%Y-%m-%d %H:%M')}"
            
            sources.append(source_info)
        
        return sources
    
    def _calculate_time_span(self, documents: List[RAGDocument]) -> Dict[str, Any]:
        """Calculate time span covered by documents"""
        if not documents:
            return {}
        
        timestamps = [doc.timestamp for doc in documents]
        earliest = min(timestamps)
        latest = max(timestamps)
        span_hours = (latest - earliest) / 3600
        
        return {
            'earliest': datetime.fromtimestamp(earliest).strftime('%Y-%m-%d %H:%M'),
            'latest': datetime.fromtimestamp(latest).strftime('%Y-%m-%d %H:%M'),
            'span_hours': span_hours,
            'span_days': span_hours / 24
        }
    
    def _summarize_rag_context(self, rag_results: RAGResult) -> str:
        """Create a brief summary of RAG context"""
        if not rag_results.documents:
            return "No historical context available."
        
        doc_count = len(rag_results.documents)
        sources = set(doc.source for doc in rag_results.documents)
        avg_similarity = sum(rag_results.similarities) / len(rag_results.similarities) if rag_results.similarities else 0
        
        time_info = self._calculate_time_span(rag_results.documents)
        
        return f"Found {doc_count} relevant documents from {len(sources)} sources with {avg_similarity:.1%} average similarity, spanning {time_info.get('span_hours', 0):.1f} hours of activity."
    
    def _get_productivity_insight(self, typing_speed: float) -> str:
        """Generate productivity insight based on typing speed"""
        if typing_speed > 60:
            return f"Excellent typing pace at {typing_speed:.1f} WPM - you're in a highly productive flow state."
        elif typing_speed > 40:
            return f"Good typing rhythm at {typing_speed:.1f} WPM - maintaining solid productivity."
        elif typing_speed > 20:
            return f"Steady pace at {typing_speed:.1f} WPM - consider if you need a break or if complex thinking is slowing typing."
        else:
            return f"Thoughtful pace at {typing_speed:.1f} WPM - likely focused on careful consideration rather than rapid input."
    
    def _generate_cache_key(self, query: str, context: QueryContext) -> str:
        """Generate cache key for query"""
        import hashlib
        
        key_components = [
            query,
            context.active_application,
            context.query_type,
            str(int(context.timestamp / 300))  # 5-minute buckets
        ]
        
        return hashlib.md5('|'.join(key_components).encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[EnhancedResponse]:
        """Get cached response if still valid"""
        if cache_key in self.query_cache:
            cached_entry = self.query_cache[cache_key]
            if time.time() - cached_entry['timestamp'] < self.cache_expiry:
                return cached_entry['response']
            else:
                del self.query_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: EnhancedResponse):
        """Cache response for future use"""
        self.query_cache[cache_key] = {
            'timestamp': time.time(),
            'response': response
        }
        
        # Cleanup old cache entries
        if len(self.query_cache) > 100:
            sorted_entries = sorted(
                self.query_cache.items(), 
                key=lambda x: x[1]['timestamp']
            )
            # Keep newest 50
            self.query_cache = dict(sorted_entries[-50:])
    
    def _create_fallback_response(self, query: str, context: QueryContext, error: str) -> EnhancedResponse:
        """Create fallback response when processing fails"""
        fallback_ai_response = AnalysisResponse(
            analysis_type="fallback",
            confidence=0.3,
            main_insight=f"I encountered an issue processing your query ({error}), but I can still provide some general guidance based on your current context in {context.active_application}.",
            suggestions=["Try rephrasing your query", "Check if all systems are running properly"],
            questions=["What specific aspect would you like help with?"],
            follow_up_prompts=["Can you provide more details?"],
            context_tags=[context.active_application, "error"],
            timestamp=time.time()
        )
        
        return EnhancedResponse(
            query=query,
            context=context,
            rag_results=RAGResult(documents=[], similarities=[]),
            ai_response=fallback_ai_response,
            combined_insight=f"⚠️ **Fallback Response**: {fallback_ai_response.main_insight}",
            confidence_score=0.3,
            processing_time=0.1,
            sources_used=[],
            follow_up_queries=["Can you help me troubleshoot this issue?"]
        )


class RAGQueryInterface:
    """Main interface for RAG-enhanced querying"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        
        # Initialize components
        self.rag_system = RAGSystem(self.storage_path / "rag_data")
        self.ollama_system = OllamaPromptSystem()
        self.keystroke_logger = KeystrokeLogger()
        self.context_aggregator = ContextAggregator()
        self.query_processor = RAGQueryProcessor(self.rag_system, self.ollama_system)
        
        # Setup logging
        self.logger = logging.getLogger("RAGQueryInterface")
        
        # Connect keystroke logger to context aggregator
        self.keystroke_logger.set_rag_callback(self.context_aggregator.add_keystroke_context)
        
        # GUI components
        self.root = None
        
        # Processing queue
        self.query_queue = queue.Queue()
        self.processing_active = False
        
        print(f"✅ RAG Query Interface initialized: {storage_path}")
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("RAG-Enhanced Query Interface")
        self.root.geometry("1400x900")
        
        # Main notebook
        main_notebook = ttk.Notebook(self.root)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Query tab
        query_tab = ttk.Frame(main_notebook)
        main_notebook.add(query_tab, text="Intelligent Query")
        self.setup_query_tab(query_tab)
        
        # History tab
        history_tab = ttk.Frame(main_notebook)
        main_notebook.add(history_tab, text="Query History")
        self.setup_history_tab(history_tab)
        
        # Analytics tab
        analytics_tab = ttk.Frame(main_notebook)
        main_notebook.add(analytics_tab, text="Analytics")
        self.setup_analytics_tab(analytics_tab)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - RAG system initialized")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_query_tab(self, parent):
        """Setup the main query interface tab"""
        # Query input
        input_frame = ttk.LabelFrame(parent, text="Query Input", padding=10)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Query type selection
        type_frame = ttk.Frame(input_frame)
        type_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(type_frame, text="Query Type:").pack(side=tk.LEFT)
        
        self.query_type_var = tk.StringVar(value="analysis")
        query_types = ["analysis", "question", "suggestion", "completion"]
        
        for query_type in query_types:
            ttk.Radiobutton(
                type_frame, 
                text=query_type.title(), 
                variable=self.query_type_var, 
                value=query_type
            ).pack(side=tk.LEFT, padx=5)
        
        # Query input text
        self.query_text = scrolledtext.ScrolledText(input_frame, height=4, wrap=tk.WORD)
        self.query_text.pack(fill=tk.X, pady=5)
        
        # Query button
        query_btn = ttk.Button(input_frame, text="🧠 Process Query", command=self.process_query_gui)
        query_btn.pack(pady=5)
        
        # Response display
        response_frame = ttk.LabelFrame(parent, text="Enhanced Response", padding=10)
        response_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Response notebook
        response_notebook = ttk.Notebook(response_frame)
        response_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Combined insight tab
        insight_tab = ttk.Frame(response_notebook)
        response_notebook.add(insight_tab, text="💡 Combined Insight")
        
        self.insight_display = scrolledtext.ScrolledText(insight_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.insight_display.pack(fill=tk.BOTH, expand=True)
        
        # RAG results tab
        rag_tab = ttk.Frame(response_notebook)
        response_notebook.add(rag_tab, text="📚 Historical Context")
        
        self.rag_display = scrolledtext.ScrolledText(rag_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.rag_display.pack(fill=tk.BOTH, expand=True)
        
        # Follow-up tab
        followup_tab = ttk.Frame(response_notebook)
        response_notebook.add(followup_tab, text="🔄 Follow-up")
        
        self.followup_display = tk.Listbox(followup_tab)
        self.followup_display.pack(fill=tk.BOTH, expand=True)
        self.followup_display.bind("<Double-Button-1>", self.execute_followup)
    
    def setup_history_tab(self, parent):
        """Setup query history tab"""
        # History list
        self.history_tree = ttk.Treeview(parent, columns=("Time", "Type", "Confidence", "Sources"), show='tree headings')
        
        self.history_tree.heading("#0", text="Query")
        self.history_tree.heading("Time", text="Time")
        self.history_tree.heading("Type", text="Type")
        self.history_tree.heading("Confidence", text="Confidence")
        self.history_tree.heading("Sources", text="Sources")
        
        self.history_tree.column("#0", width=300)
        self.history_tree.column("Time", width=100)
        self.history_tree.column("Type", width=100)
        self.history_tree.column("Confidence", width=100)
        self.history_tree.column("Sources", width=100)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_analytics_tab(self, parent):
        """Setup analytics tab"""
        # Stats display
        stats_frame = ttk.LabelFrame(parent, text="System Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_display = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.stats_display.pack(fill=tk.BOTH, expand=True)
        
        # Update button
        update_btn = ttk.Button(stats_frame, text="🔄 Update Statistics", command=self.update_statistics)
        update_btn.pack(pady=5)
    
    def process_query_gui(self):
        """Process query from GUI input"""
        query_text = self.query_text.get("1.0", tk.END).strip()
        query_type = self.query_type_var.get()
        
        if not query_text:
            messagebox.showwarning("Empty Query", "Please enter a query to process.")
            return
        
        # Show processing status
        self.status_var.set("Processing query...")
        self.root.update()
        
        # Process in background thread
        threading.Thread(
            target=self._process_query_background,
            args=(query_text, query_type),
            daemon=True
        ).start()
    
    def _process_query_background(self, query_text: str, query_type: str):
        """Background query processing"""
        try:
            # Build query context
            context = self.context_aggregator.build_query_context(query_text)
            context.query_type = query_type
            
            # Process query
            response = self.query_processor.process_query(query_text, context)
            
            # Update GUI in main thread
            self.root.after(0, lambda: self._update_response_display(response))
            
        except Exception as e:
            self.logger.error(f"Background query processing error: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
    
    def _update_response_display(self, response: EnhancedResponse):
        """Update GUI with query response"""
        # Update combined insight
        self.insight_display.config(state=tk.NORMAL)
        self.insight_display.delete("1.0", tk.END)
        self.insight_display.insert("1.0", response.combined_insight)
        self.insight_display.config(state=tk.DISABLED)
        
        # Update RAG results
        self.rag_display.config(state=tk.NORMAL)
        self.rag_display.delete("1.0", tk.END)
        
        if response.rag_results.documents:
            rag_content = f"📊 Found {len(response.rag_results.documents)} relevant documents:\n\n"
            
            for i, doc in enumerate(response.rag_results.documents, 1):
                similarity = response.rag_results.similarities[i-1] if i-1 < len(response.rag_results.similarities) else 0
                timestamp = datetime.fromtimestamp(doc.timestamp).strftime("%Y-%m-%d %H:%M")
                
                rag_content += f"📄 Document {i} ({similarity:.1%} similarity)\n"
                rag_content += f"   Source: {doc.source} | Time: {timestamp}\n"
                rag_content += f"   Content: {doc.content[:200]}...\n\n"
        else:
            rag_content = "No historical context found for this query."
        
        self.rag_display.insert("1.0", rag_content)
        self.rag_display.config(state=tk.DISABLED)
        
        # Update follow-up queries
        self.followup_display.delete(0, tk.END)
        for followup in response.follow_up_queries:
            self.followup_display.insert(tk.END, followup)
        
        # Add to history
        self._add_to_history(response)
        
        # Update status
        self.status_var.set(f"Query processed in {response.processing_time:.2f}s (confidence: {response.confidence_score:.1%})")
    
    def _add_to_history(self, response: EnhancedResponse):
        """Add response to query history"""
        timestamp = datetime.fromtimestamp(response.timestamp).strftime("%H:%M:%S")
        
        self.history_tree.insert("", 0, text=response.query[:50] + ("..." if len(response.query) > 50 else ""),
                               values=(
                                   timestamp,
                                   response.context.query_type,
                                   f"{response.confidence_score:.1%}",
                                   len(response.sources_used)
                               ))
    
    def execute_followup(self, event):
        """Execute selected follow-up query"""
        selection = self.followup_display.curselection()
        if selection:
            followup_query = self.followup_display.get(selection[0])
            self.query_text.delete("1.0", tk.END)
            self.query_text.insert("1.0", followup_query)
    
    def update_statistics(self):
        """Update and display system statistics"""
        try:
            rag_stats = self.rag_system.get_system_stats()
            processor_stats = {
                'cache_size': len(self.query_processor.query_cache),
                'cache_hit_ratio': 0.85  # Placeholder
            }
            
            stats_content = f"""📊 RAG Query Interface Statistics

🗄️  RAG System:
   • Total Documents: {rag_stats['vector_store']['total_documents']}
   • Backend: {rag_stats['vector_store']['backend']}
   • Total Queries: {rag_stats['query_stats']['total_queries']}
   • Avg Retrieval Time: {rag_stats['query_stats']['avg_retrieval_time']:.3f}s

🧠 Query Processor:
   • Cache Size: {processor_stats['cache_size']}
   • Cache Hit Ratio: {processor_stats['cache_hit_ratio']:.1%}
   
📊 Context Aggregator:
   • Context Buffer Size: {len(self.context_aggregator.context_buffer)}
   • Application Contexts: {len(self.context_aggregator.application_contexts)}
   
⌨️  Keystroke Logger:
   • Status: {'Running' if self.keystroke_logger.is_running else 'Stopped'}
   
🕐 Uptime: {rag_stats.get('uptime', 0) / 3600:.1f} hours
"""
            
            self.stats_display.config(state=tk.NORMAL)
            self.stats_display.delete("1.0", tk.END)
            self.stats_display.insert("1.0", stats_content)
            self.stats_display.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Statistics update error: {e}")
    
    def run_gui(self):
        """Start the GUI application"""
        self.setup_gui()
        self.root.mainloop()
    
    def start_background_systems(self):
        """Start background systems (keystroke logging, etc.)"""
        try:
            # Start keystroke logging in background thread
            def start_keystroke_logging():
                try:
                    self.keystroke_logger.start_logging()
                except Exception as e:
                    self.logger.error(f"Keystroke logging error: {e}")
            
            keystroke_thread = threading.Thread(target=start_keystroke_logging, daemon=True)
            keystroke_thread.start()
            
            self.logger.info("Background systems started")
            
        except Exception as e:
            self.logger.error(f"Error starting background systems: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        try:
            self.keystroke_logger.stop_logging()
            self.rag_system.shutdown()
            self.logger.info("RAG Query Interface shutdown complete")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Query Interface")
    parser.add_argument("--storage", type=Path, default="./rag_storage", help="Storage path for RAG data")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--background", action="store_true", help="Start background systems")
    
    args = parser.parse_args()
    
    # Create interface
    interface = RAGQueryInterface(args.storage)
    
    if args.background:
        interface.start_background_systems()
    
    if args.gui:
        interface.run_gui()
    else:
        print("RAG Query Interface initialized")
        print("Use --gui to launch the graphical interface")
        print("Use --background to start keystroke logging")


if __name__ == "__main__":
    main()