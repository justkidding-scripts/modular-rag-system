#!/usr/bin/env python3
"""
Real-time Keystroke Logging System with Privacy Controls
Securely captures keyboard input for RAG system integration with comprehensive privacy features
"""

import time
import json
import threading
import hashlib
import re
from typing import Dict, List, Optional, Set, Callable, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
import queue
import signal
import sys

# Try to import platform-specific keyboard libraries
try:
    from pynput import keyboard
    from pynput.keyboard import Key, Listener
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("pynput not available - install with: pip install pynput")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - install with: pip install psutil")


@dataclass
class KeystrokeEvent:
    """Represents a keystroke event with context"""
    timestamp: float
    key: str
    key_type: str  # 'char', 'special', 'modifier'
    is_pressed: bool
    window_title: Optional[str] = None
    application: Optional[str] = None
    session_id: str = ""
    filtered: bool = False
    anonymized: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TypingSession:
    """Represents a typing session with aggregated data"""
    session_id: str
    start_time: float
    end_time: float
    application: str
    window_title: str
    keystroke_count: int
    word_count: int
    char_count: int
    typing_speed: float  # WPM
    content_hash: str
    filtered_content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PrivacyFilter:
    """Handles privacy filtering and data anonymization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sensitive_patterns = self._load_sensitive_patterns()
        self.blocked_applications = set(config.get('blocked_applications', []))
        self.blocked_window_titles = set(config.get('blocked_window_titles', []))
        
        # Password/sensitive input detection
        self.password_indicators = {
            'password', 'passwd', 'pass', 'login', 'auth', 'credential',
            'secret', 'key', 'token', 'pin', 'security', 'private'
        }
        
        # Common sensitive data patterns
        self.sensitive_regex = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',                        # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email (sometimes)
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',  # Phone
            r'\b[A-Z]{2}\d{6}[A-Z]?\b',                      # Passport-like patterns
            r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)[:=]\s*[^\s]+',  # API keys
        ]
        
        self.compiled_patterns = [re.compile(pattern) for pattern in self.sensitive_regex]
        
    def _load_sensitive_patterns(self) -> List[str]:
        """Load additional sensitive patterns from config"""
        return self.config.get('sensitive_patterns', [])
    
    def should_block_application(self, app_name: str, window_title: str) -> bool:
        """Check if application should be blocked from logging"""
        if not self.config.get('enable_app_filtering', True):
            return False
        
        app_lower = app_name.lower() if app_name else ""
        title_lower = window_title.lower() if window_title else ""
        
        # Check blocked applications
        for blocked_app in self.blocked_applications:
            if blocked_app.lower() in app_lower:
                return True
        
        # Check blocked window titles
        for blocked_title in self.blocked_window_titles:
            if blocked_title.lower() in title_lower:
                return True
        
        # Check for password fields or sensitive contexts
        if any(indicator in title_lower for indicator in self.password_indicators):
            return True
        
        return False
    
    def is_sensitive_content(self, text: str) -> bool:
        """Check if text contains sensitive information"""
        if not self.config.get('enable_content_filtering', True):
            return False
        
        text_lower = text.lower()
        
        # Check for password indicators
        if any(indicator in text_lower for indicator in self.password_indicators):
            return True
        
        # Check regex patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        
        # Check custom patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize sensitive parts of text"""
        if not self.config.get('enable_anonymization', False):
            return text
        
        anonymized = text
        
        # Replace sensitive patterns with placeholders
        for i, pattern in enumerate(self.compiled_patterns):
            anonymized = pattern.sub(f'[REDACTED_{i}]', anonymized)
        
        return anonymized
    
    def filter_keystroke_sequence(self, keystrokes: List[KeystrokeEvent]) -> List[KeystrokeEvent]:
        """Filter a sequence of keystrokes for privacy"""
        if not keystrokes:
            return []
        
        filtered = []
        current_text = ""
        
        for keystroke in keystrokes:
            # Skip if application is blocked
            if self.should_block_application(keystroke.application or "", keystroke.window_title or ""):
                keystroke.filtered = True
                continue
            
            # Build current text context
            if keystroke.key_type == 'char' and keystroke.is_pressed:
                current_text += keystroke.key
            
            # Check for sensitive content in accumulated text
            if len(current_text) > 10 and self.is_sensitive_content(current_text):
                # Mark recent keystrokes as filtered
                for recent_keystroke in filtered[-10:]:
                    recent_keystroke.filtered = True
                keystroke.filtered = True
                current_text = ""  # Reset
                continue
            
            filtered.append(keystroke)
        
        return filtered


class WindowTracker:
    """Tracks active window and application context"""
    
    def __init__(self):
        self.current_window = ""
        self.current_app = ""
        self.window_history = deque(maxlen=50)
        
    def get_active_window_info(self) -> tuple[str, str]:
        """Get current active window title and application"""
        try:
            if PSUTIL_AVAILABLE:
                # Get active window using psutil (Linux-specific approach)
                return self._get_linux_window_info()
            else:
                return "Unknown", "Unknown"
        except Exception as e:
            return "Error", f"Error: {e}"
    
    def _get_linux_window_info(self) -> tuple[str, str]:
        """Get window info on Linux systems"""
        try:
            import subprocess
            
            # Try to get active window using xdotool
            try:
                window_id = subprocess.check_output(
                    ['xdotool', 'getactivewindow'], 
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                
                window_title = subprocess.check_output(
                    ['xdotool', 'getwindowname', window_id],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                
                # Get application name
                app_name = subprocess.check_output(
                    ['xdotool', 'getwindowclassname', window_id],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                
                return window_title, app_name
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: try wmctrl
                try:
                    wmctrl_output = subprocess.check_output(
                        ['wmctrl', '-l'], 
                        stderr=subprocess.DEVNULL
                    ).decode()
                    
                    lines = wmctrl_output.strip().split('\n')
                    if lines:
                        # Get the first active window (basic approach)
                        parts = lines[0].split(None, 3)
                        if len(parts) >= 4:
                            return parts[3], "Unknown"
                    
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
        
        except Exception:
            pass
        
        return "Unknown", "Unknown"
    
    def update_window_context(self):
        """Update current window context"""
        title, app = self.get_active_window_info()
        
        if title != self.current_window or app != self.current_app:
            self.window_history.append({
                'timestamp': time.time(),
                'previous_window': self.current_window,
                'previous_app': self.current_app,
                'new_window': title,
                'new_app': app
            })
            
            self.current_window = title
            self.current_app = app
    
    def get_current_context(self) -> Dict[str, str]:
        """Get current window context"""
        return {
            'window_title': self.current_window,
            'application': self.current_app
        }


class KeystrokeLogger:
    """Main keystroke logging system with privacy controls"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.privacy_filter = PrivacyFilter(self.config['privacy'])
        self.window_tracker = WindowTracker()
        
        # Logging setup
        self.setup_logging()
        
        # Data storage
        self.keystroke_buffer = deque(maxlen=10000)
        self.session_buffer = deque(maxlen=1000)
        self.current_session = None
        
        # Processing
        self.processing_queue = queue.Queue(maxsize=1000)
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_keystrokes': 0,
            'filtered_keystrokes': 0,
            'sessions_created': 0,
            'start_time': time.time()
        }
        
        # Callback for RAG integration
        self.rag_callback = None
        
        # Session management
        self.session_timeout = self.config.get('session_timeout', 300)  # 5 minutes
        self.last_activity_time = time.time()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'logging': {
                'log_level': 'INFO',
                'log_file': 'keystroke_logger.log',
                'max_log_size': 10 * 1024 * 1024  # 10MB
            },
            'privacy': {
                'enable_app_filtering': True,
                'enable_content_filtering': True,
                'enable_anonymization': True,
                'blocked_applications': [
                    'password', 'keepass', 'lastpass', 'bitwarden',
                    'gnome-keyring', 'kwallet', 'seahorse',
                    'banking', 'credit', 'paypal'
                ],
                'blocked_window_titles': [
                    'password', 'login', 'sign in', 'authenticate',
                    'security', 'private', 'confidential'
                ],
                'sensitive_patterns': []
            },
            'processing': {
                'buffer_size': 10000,
                'session_timeout': 300,
                'min_session_keystrokes': 10,
                'save_interval': 60
            },
            'features': {
                'track_typing_speed': True,
                'detect_sessions': True,
                'export_sessions': True,
                'rag_integration': True
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge configurations
                    for key, value in default_config.items():
                        if key in loaded_config and isinstance(value, dict):
                            default_config[key].update(loaded_config[key])
                        elif key in loaded_config:
                            default_config[key] = loaded_config[key]
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def setup_logging(self):
        """Setup logging system"""
        self.logger = logging.getLogger("KeystrokeLogger")
        self.logger.setLevel(getattr(logging, self.config['logging']['log_level']))
        
        # File handler with rotation
        log_file = Path(self.config['logging']['log_file'])
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def set_rag_callback(self, callback: Callable[[str, Dict], None]):
        """Set callback function for RAG system integration"""
        self.rag_callback = callback
        self.logger.info("RAG callback registered")
    
    def on_key_press(self, key):
        """Handle key press events"""
        self._process_key_event(key, True)
    
    def on_key_release(self, key):
        """Handle key release events"""
        self._process_key_event(key, False)
    
    def _process_key_event(self, key, is_pressed: bool):
        """Process individual key events"""
        try:
            # Update window context periodically
            current_time = time.time()
            if current_time - getattr(self, '_last_window_update', 0) > 1.0:
                self.window_tracker.update_window_context()
                self._last_window_update = current_time
            
            # Determine key information
            key_str, key_type = self._parse_key(key)
            
            # Create keystroke event
            context = self.window_tracker.get_current_context()
            
            event = KeystrokeEvent(
                timestamp=current_time,
                key=key_str,
                key_type=key_type,
                is_pressed=is_pressed,
                window_title=context['window_title'],
                application=context['application'],
                session_id=self._get_current_session_id()
            )
            
            # Add to processing queue
            try:
                self.processing_queue.put_nowait(event)
                self.stats['total_keystrokes'] += 1
                self.last_activity_time = current_time
            except queue.Full:
                self.logger.warning("Processing queue full, dropping keystroke")
            
        except Exception as e:
            self.logger.error(f"Error processing key event: {e}")
    
    def _parse_key(self, key) -> tuple[str, str]:
        """Parse key object into string representation and type"""
        try:
            if hasattr(key, 'char') and key.char is not None:
                return key.char, 'char'
            elif hasattr(key, 'name'):
                return key.name, 'special'
            else:
                return str(key), 'modifier'
        except AttributeError:
            return str(key), 'unknown'
    
    def _get_current_session_id(self) -> str:
        """Get or create current session ID"""
        current_time = time.time()
        context = self.window_tracker.get_current_context()
        
        # Check if we need a new session
        if (self.current_session is None or 
            current_time - self.last_activity_time > self.session_timeout or
            self.current_session.application != context['application']):
            
            self._start_new_session(context)
        
        return self.current_session.session_id if self.current_session else "default"
    
    def _start_new_session(self, context: Dict[str, str]):
        """Start a new typing session"""
        session_id = hashlib.md5(
            f"{context['application']}_{context['window_title']}_{time.time()}".encode()
        ).hexdigest()[:12]
        
        self.current_session = TypingSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=0.0,
            application=context['application'],
            window_title=context['window_title'],
            keystroke_count=0,
            word_count=0,
            char_count=0,
            typing_speed=0.0,
            content_hash="",
            filtered_content="",
            metadata={}
        )
        
        self.stats['sessions_created'] += 1
        self.logger.info(f"Started new session: {session_id} for {context['application']}")
    
    def _process_keystroke_buffer(self):
        """Background processing of keystroke buffer"""
        while self.is_running:
            try:
                # Get keystroke from queue
                event = self.processing_queue.get(timeout=1.0)
                
                # Apply privacy filtering
                if self.privacy_filter.should_block_application(
                    event.application or "", 
                    event.window_title or ""
                ):
                    event.filtered = True
                    self.stats['filtered_keystrokes'] += 1
                    continue
                
                # Add to buffer
                self.keystroke_buffer.append(event)
                
                # Update current session
                if self.current_session and event.session_id == self.current_session.session_id:
                    self.current_session.keystroke_count += 1
                    if event.key_type == 'char':
                        self.current_session.char_count += 1
                
                # Process accumulated keystrokes periodically
                if len(self.keystroke_buffer) >= 100:
                    self._process_accumulated_keystrokes()
                
            except queue.Empty:
                # Timeout - process any accumulated keystrokes
                if self.keystroke_buffer:
                    self._process_accumulated_keystrokes()
                continue
            except Exception as e:
                self.logger.error(f"Error in processing thread: {e}")
    
    def _process_accumulated_keystrokes(self):
        """Process accumulated keystrokes for content analysis"""
        if not self.keystroke_buffer:
            return
        
        # Group keystrokes by session
        sessions = defaultdict(list)
        for keystroke in list(self.keystroke_buffer):
            if not keystroke.filtered:
                sessions[keystroke.session_id].append(keystroke)
        
        # Process each session
        for session_id, keystrokes in sessions.items():
            self._analyze_typing_session(session_id, keystrokes)
        
        # Clear processed keystrokes
        self.keystroke_buffer.clear()
    
    def _analyze_typing_session(self, session_id: str, keystrokes: List[KeystrokeEvent]):
        """Analyze typing session for patterns and content"""
        if not keystrokes:
            return
        
        # Build text content from keystrokes
        content = self._reconstruct_text(keystrokes)
        
        # Apply privacy filtering to content
        if self.privacy_filter.is_sensitive_content(content):
            self.logger.info(f"Filtered sensitive content from session {session_id}")
            return
        
        # Anonymize if configured
        if self.config['privacy']['enable_anonymization']:
            content = self.privacy_filter.anonymize_text(content)
        
        # Calculate metrics
        word_count = len(content.split())
        char_count = len(content)
        
        # Calculate typing speed
        if keystrokes:
            time_span = keystrokes[-1].timestamp - keystrokes[0].timestamp
            typing_speed = (word_count / (time_span / 60)) if time_span > 0 else 0
        else:
            typing_speed = 0
        
        # Update session data
        if self.current_session and self.current_session.session_id == session_id:
            self.current_session.word_count = word_count
            self.current_session.char_count = char_count
            self.current_session.typing_speed = typing_speed
            self.current_session.content_hash = hashlib.md5(content.encode()).hexdigest()
            self.current_session.filtered_content = content[:500]  # Truncate for storage
            self.current_session.end_time = time.time()
        
        # Send to RAG system if enabled and callback is set
        if (self.config['features']['rag_integration'] and 
            self.rag_callback and 
            len(content.strip()) >= 10):
            
            metadata = {
                'session_id': session_id,
                'application': keystrokes[0].application,
                'window_title': keystrokes[0].window_title,
                'keystroke_count': len(keystrokes),
                'word_count': word_count,
                'typing_speed': typing_speed,
                'timestamp': keystrokes[0].timestamp
            }
            
            try:
                self.rag_callback(content, metadata)
                self.logger.info(f"Sent session data to RAG: {len(content)} chars")
            except Exception as e:
                self.logger.error(f"RAG callback error: {e}")
    
    def _reconstruct_text(self, keystrokes: List[KeystrokeEvent]) -> str:
        """Reconstruct text from keystroke events"""
        text = ""
        
        for keystroke in keystrokes:
            if not keystroke.is_pressed:
                continue
                
            if keystroke.key_type == 'char':
                text += keystroke.key
            elif keystroke.key_type == 'special':
                if keystroke.key == 'space':
                    text += ' '
                elif keystroke.key == 'enter':
                    text += '\n'
                elif keystroke.key == 'tab':
                    text += '\t'
                elif keystroke.key == 'backspace' and text:
                    text = text[:-1]
        
        return text
    
    def start_logging(self):
        """Start the keystroke logging system"""
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput is required for keystroke logging")
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_keystroke_buffer, daemon=True)
        self.processing_thread.start()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Starting keystroke logging...")
        
        # Start keyboard listener
        try:
            with Listener(on_press=self.on_key_press, on_release=self.on_key_release) as listener:
                self.logger.info("Keystroke logging started successfully")
                listener.join()
        except Exception as e:
            self.logger.error(f"Failed to start keyboard listener: {e}")
            raise
    
    def stop_logging(self):
        """Stop the keystroke logging system"""
        self.is_running = False
        
        # Process remaining keystrokes
        self._process_accumulated_keystrokes()
        
        # Finalize current session
        if self.current_session:
            self.current_session.end_time = time.time()
            self.session_buffer.append(self.current_session)
        
        self.logger.info("Keystroke logging stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down keystroke logger...")
        self.stop_logging()
        sys.exit(0)
    
    def export_sessions(self, export_path: Path) -> bool:
        """Export typing sessions to file"""
        try:
            export_data = {
                'timestamp': time.time(),
                'stats': self.stats,
                'sessions': [session.to_dict() for session in self.session_buffer]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Sessions exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        uptime = time.time() - self.stats['start_time']
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'keystrokes_per_minute': self.stats['total_keystrokes'] / max(uptime / 60, 1),
            'filter_ratio': self.stats['filtered_keystrokes'] / max(self.stats['total_keystrokes'], 1),
            'current_session': self.current_session.to_dict() if self.current_session else None,
            'session_count': len(self.session_buffer),
            'buffer_size': len(self.keystroke_buffer)
        }


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Privacy-aware Keystroke Logger")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--export", type=Path, help="Export sessions to file")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    logger = KeystrokeLogger(args.config)
    
    if args.stats:
        stats = logger.get_statistics()
        print(json.dumps(stats, indent=2))
        return
    
    if args.export:
        success = logger.export_sessions(args.export)
        print(f"Export {'succeeded' if success else 'failed'}")
        return
    
    # Start logging
    try:
        print("🔒 Privacy-aware keystroke logging started")
        print("📊 Privacy features enabled:")
        print(f"   ✅ Application filtering: {logger.config['privacy']['enable_app_filtering']}")
        print(f"   ✅ Content filtering: {logger.config['privacy']['enable_content_filtering']}")
        print(f"   ✅ Data anonymization: {logger.config['privacy']['enable_anonymization']}")
        print("⚠️  Press Ctrl+C to stop")
        
        logger.start_logging()
    except KeyboardInterrupt:
        print("\nStopping keystroke logger...")
        logger.stop_logging()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()