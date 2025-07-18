#!/usr/bin/env python3
"""
Ollama Chat Utility for Affinify
Provides chat interface with streaming support for the AI Assistant
"""

import json
import streamlit as st
import requests
from pathlib import Path
import time
from typing import Generator, Optional, Dict, Any

class OllamaChat:
    """Chat interface for Ollama with streaming support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.host = config.get('host', 'http://localhost:11434')
        self.model = config.get('model', 'llama3.2:3b')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
        self.timeout = config.get('timeout', 30)
        self.system_prompt = config.get('system_prompt', '')
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
            # Add welcome message
            welcome_msg = self.config.get('welcome_message', 'Hello! How can I help you today?')
            st.session_state.chat_messages.append({
                'role': 'assistant',
                'content': welcome_msg
            })
        
        if 'ollama_available' not in st.session_state:
            st.session_state.ollama_available = self.check_ollama_availability()
    
    def check_ollama_availability(self) -> bool:
        """Check if Ollama is available and model is accessible"""
        if not self.enabled:
            return False
            
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Check if exact model or base model is available
            if self.model in model_names:
                return True
            
            # Check if base model name matches (e.g., "llama3.2" matches "llama3.2:3b")
            base_model = self.model.split(':')[0]
            for model_name in model_names:
                if model_name.startswith(base_model):
                    return True
            
            return False
            
        except Exception as e:
            print(f"Ollama availability check failed: {e}")
            return False
    
    def send_message_stream(self, message: str) -> Generator[str, None, None]:
        """Send message to Ollama and yield streaming response"""
        if not self.enabled or not st.session_state.ollama_available:
            yield "AI Assistant is not available."
            return
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": self._build_prompt(message),
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Send streaming request
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                yield data['response']
                            
                            # Check if this is the final response
                            if data.get('done', False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: Received status code {response.status_code}"
                
        except requests.RequestException as e:
            yield f"Connection error: {str(e)}"
        except Exception as e:
            yield f"Unexpected error: {str(e)}"
    
    def send_message(self, message: str) -> Optional[str]:
        """Send message to Ollama and get complete response (non-streaming)"""
        if not self.enabled or not st.session_state.ollama_available:
            return "AI Assistant is not available."
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": self._build_prompt(message),
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Send request
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '')
            else:
                return f"Error: Received status code {response.status_code}"
                
        except requests.RequestException as e:
            return f"Connection error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def _build_prompt(self, message: str) -> str:
        """Build the complete prompt with system message and conversation history"""
        prompt_parts = []
        
        # Add system prompt if available
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")
        
        # Add conversation history (last 10 messages to keep context manageable)
        recent_messages = st.session_state.chat_messages[-10:]
        for msg in recent_messages:
            if msg['role'] == 'user':
                prompt_parts.append(f"Human: {msg['content']}")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        # Add current message
        prompt_parts.append(f"Human: {message}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def add_message(self, role: str, content: str):
        """Add message to chat history"""
        st.session_state.chat_messages.append({
            'role': role,
            'content': content
        })
    
    def clear_chat(self):
        """Clear chat history"""
        st.session_state.chat_messages = []
        # Add welcome message back
        welcome_msg = self.config.get('welcome_message', 'Hello! How can I help you today?')
        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': welcome_msg
        })
    
    def get_chat_history(self) -> list:
        """Get current chat history"""
        return st.session_state.chat_messages.copy()


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('ollama', {})
        else:
            print(f"Config file not found at {config_path}")
            return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def create_ollama_chat() -> OllamaChat:
    """Create and return an OllamaChat instance"""
    config = load_config()
    return OllamaChat(config)


# For testing purposes
if __name__ == "__main__":
    chat = create_ollama_chat()
    
    if chat.enabled:
        print("Ollama Chat initialized successfully!")
        print(f"Model: {chat.model}")
        print(f"Host: {chat.host}")
        print(f"Available: {chat.check_ollama_availability()}")
        
        # Test streaming
        print("\nTesting streaming response:")
        for chunk in chat.send_message_stream("Hello, how are you?"):
            print(chunk, end='', flush=True)
        print("\n")
        
    else:
        print("Ollama Chat is disabled in configuration.")