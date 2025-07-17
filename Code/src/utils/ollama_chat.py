#!/usr/bin/env python3
"""
Ollama Chat Integration for Affinify
Provides AI assistant functionality using Ollama
"""

import requests
import json
import time
import streamlit as st
from typing import Optional, Dict, Any, List
import logging

class OllamaChat:
    """Ollama chat integration for Affinify Assistant"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.host = config.get('host', 'http://localhost:11434')
        self.model = config.get('model', 'llama3.2:3b')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
        self.system_prompt = config.get('system_prompt', 'You are a helpful AI assistant.')
        self.timeout = config.get('timeout', 30)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state for chat
        self.init_chat_session()
    
    def init_chat_session(self):
        """Initialize chat session state"""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        if 'chat_initialized' not in st.session_state:
            st.session_state.chat_initialized = False
            
        if 'ollama_available' not in st.session_state:
            st.session_state.ollama_available = self.check_ollama_availability()
    
    def check_ollama_availability(self) -> bool:
        """Check if Ollama is available and the model is accessible"""
        if not self.enabled:
            return False
            
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if our model is available
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model in model_names:
                    return True
                else:
                    self.logger.warning(f"Model {self.model} not found in available models: {model_names}")
                    return False
            else:
                self.logger.error(f"Ollama not accessible: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking Ollama availability: {e}")
            return False
    
    def send_message(self, message: str, context: Optional[str] = None) -> Optional[str]:
        """Send message to Ollama and get response"""
        if not self.enabled or not st.session_state.ollama_available:
            return None
        
        try:
            # Prepare the prompt with just the system prompt and current message
            full_prompt = self.system_prompt
            
            # Add context if provided
            if context:
                full_prompt += f"\n\nContext: {context}"
            
            # Add the current message directly
            full_prompt += f"\n\nHuman: {message}\nAssistant:"
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error sending message to Ollama: {e}")
            return None
    
    def get_streaming_response(self, message: str, context: Optional[str] = None):
        """Get streaming response from Ollama"""
        if not self.enabled or not st.session_state.ollama_available:
            return None
        
        try:
            # Prepare the prompt
            full_prompt = self.system_prompt
            
            if context:
                full_prompt += f"\n\nContext: {context}"
            
            if st.session_state.chat_messages:
                full_prompt += "\n\nChat History:"
                for msg in st.session_state.chat_messages[-5:]:
                    role = "Human" if msg['role'] == 'user' else "Assistant"
                    full_prompt += f"\n{role}: {msg['content']}"
            
            full_prompt += f"\n\nHuman: {message}\nAssistant:"
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
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
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']
                        except json.JSONDecodeError:
                            continue
            else:
                self.logger.error(f"Ollama streaming API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting streaming response: {e}")
            return None
    
    def add_message(self, role: str, content: str):
        """Add message to chat history"""
        st.session_state.chat_messages.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
        
        # Keep only last 50 messages to prevent memory issues
        if len(st.session_state.chat_messages) > 50:
            st.session_state.chat_messages = st.session_state.chat_messages[-50:]
    
    def clear_chat(self):
        """Clear chat history"""
        st.session_state.chat_messages = []
        st.session_state.chat_initialized = False
    
    def get_context_from_page(self, page_name: str) -> str:
        """Get context information based on current page"""
        contexts = {
            "🏠 Home": "User is on the home page viewing system overview and getting started information.",
            "📊 Data Pipeline": "User is on the data pipeline page working with data processing, feature extraction, and model training.",
            "🔬 Predictions": "User is on the predictions page making binding affinity predictions for protein-ligand pairs.",
            "📈 Analysis": "User is on the analysis page reviewing model performance and results.",
            "⚙️ Settings": "User is on the settings page configuring system parameters and maintenance.",
            "ℹ️ About": "User is on the about page learning about the project and its technical details."
        }
        
        return contexts.get(page_name, "User is using the Affinify platform.")
    
    def render_chat_interface(self, page_name: str = ""):
        """Render the chat interface in the sidebar"""
        if not self.enabled:
            return
        
        # Chat header
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {self.config.get('chat_title', '🤖 AI Assistant')}")
        
        # Check availability
        if not st.session_state.ollama_available:
            st.sidebar.error("🔴 Ollama not available")
            st.sidebar.markdown("Please ensure Ollama is running and the model is installed.")
            
            if st.sidebar.button("🔄 Retry Connection"):
                st.session_state.ollama_available = self.check_ollama_availability()
                st.rerun()
            return
        
        # Status indicator
        st.sidebar.success("🟢 AI Assistant Ready")
        
        # Chat container
        chat_container = st.sidebar.container()
        
        # Display chat messages
        with chat_container:
            # Welcome message
            if not st.session_state.chat_initialized:
                welcome_msg = self.config.get('welcome_message', 'Hello! How can I help you?')
                st.sidebar.markdown(f"💬 **Assistant**: {welcome_msg}")
                st.session_state.chat_initialized = True
            
            # Display chat history
            for message in st.session_state.chat_messages[-10:]:  # Show last 10 messages
                if message['role'] == 'user':
                    st.sidebar.markdown(f"👤 **You**: {message['content']}")
                else:
                    st.sidebar.markdown(f"🤖 **Assistant**: {message['content']}")
        
        # Chat input
        user_input = st.sidebar.text_input(
            "Ask me anything about Affinify...",
            key="chat_input",
            placeholder="Type your question here..."
        )
        
        # Send button
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            send_button = st.button("Send", key="send_chat", use_container_width=True)
        
        with col2:
            if st.button("🗑️", key="clear_chat", help="Clear chat"):
                self.clear_chat()
                st.rerun()
        
        # Handle message sending
        if send_button and user_input.strip():
            # Add user message
            self.add_message('user', user_input)
            
            # Get page context
            context = self.get_context_from_page(page_name)
            
            # Show thinking indicator
            with st.spinner("🤔 Thinking..."):
                # Get response from Ollama
                response = self.send_message(user_input, context)
                
                if response:
                    # Add assistant response
                    self.add_message('assistant', response)
                else:
                    # Show error
                    error_msg = self.config.get('error_message', 'Sorry, I encountered an error.')
                    self.add_message('assistant', error_msg)
            
            # Refresh to show new messages
            st.rerun()
    
    def render_floating_chat(self, page_name: str = ""):
        """Render floating chat widget (alternative to sidebar)"""
        if not self.enabled or not st.session_state.ollama_available:
            return
        
        # Add floating chat CSS
        st.markdown("""
        <style>
        .floating-chat {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            height: 500px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 1000;
            display: none;
        }
        .chat-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            z-index: 1001;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Chat toggle button
        if st.button("💬", key="floating_chat_toggle", help="Open AI Assistant"):
            st.session_state.show_floating_chat = not st.session_state.get('show_floating_chat', False)
    
    def get_suggestions(self, page_name: str) -> List[str]:
        """Get suggested questions based on current page"""
        suggestions = {
            "🏠 Home": [
                "What is Affinify?",
                "How do I get started?",
                "What models are available?",
                "How accurate are the predictions?"
            ],
            "📊 Data Pipeline": [
                "How do I process BindingDB data?",
                "What features are extracted?",
                "How long does training take?",
                "What's the difference between sample and real data?"
            ],
            "🔬 Predictions": [
                "How do I make a prediction?",
                "What's a good binding affinity score?",
                "How do I interpret results?",
                "Can I predict multiple compounds?"
            ],
            "📈 Analysis": [
                "How do I interpret R² scores?",
                "Which model performs best?",
                "What does RMSE mean?",
                "How can I improve model performance?"
            ],
            "⚙️ Settings": [
                "How do I configure the system?",
                "What are the default parameters?",
                "How do I clear cached data?",
                "Where are models stored?"
            ],
            "ℹ️ About": [
                "Who created Affinify?",
                "What technology stack is used?",
                "How does binding affinity prediction work?",
                "What are the scientific applications?"
            ]
        }
        
        return suggestions.get(page_name, [
            "What is protein-ligand binding?",
            "How do machine learning models work?",
            "What is molecular modeling?",
            "How can I learn more about this field?"
        ])
    
    def render_suggestions(self, page_name: str = ""):
        """Render suggested questions"""
        if not self.enabled or not st.session_state.ollama_available:
            return
        
        suggestions = self.get_suggestions(page_name)
        
        st.sidebar.markdown("#### 💡 Suggested Questions")
        
        for i, suggestion in enumerate(suggestions[:3]):  # Show top 3 suggestions
            if st.sidebar.button(f"💬 {suggestion}", key=f"suggestion_{i}"):
                # Add suggestion as user message
                self.add_message('user', suggestion)
                
                # Get context and response
                context = self.get_context_from_page(page_name)
                
                with st.spinner("🤔 Thinking..."):
                    response = self.send_message(suggestion, context)
                    
                    if response:
                        self.add_message('assistant', response)
                    else:
                        error_msg = self.config.get('error_message', 'Sorry, I encountered an error.')
                        self.add_message('assistant', error_msg)
                
                st.rerun()


def load_ollama_config() -> Dict[str, Any]:
    """Load Ollama configuration from config file"""
    try:
        import os
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up 2 levels from src/utils
        config_path = os.path.join(project_root, 'config', 'config.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config.get('ollama', {})
        
    except Exception as e:
        logging.error(f"Error loading Ollama config: {e}")
        return {
            'enabled': False,
            'host': 'http://localhost:11434',
            'model': 'llama3.2:3b',
            'temperature': 0.7,
            'max_tokens': 1000,
            'system_prompt': 'You are a helpful AI assistant.',
            'timeout': 30
        }


def create_ollama_chat() -> OllamaChat:
    """Create and return OllamaChat instance"""
    config = load_ollama_config()
    return OllamaChat(config)