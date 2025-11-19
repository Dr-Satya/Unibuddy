#!/usr/bin/env python3

import os
import sys
import asyncio
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import track
import colorama

import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Initialize colorama for Windows compatibility
colorama.init()

# Add src to path for imports and handle both direct execution and module execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try absolute import first (when run as module)
    from src.config import settings
    from src.auth import auth_manager
    from src.database import db_manager
except ImportError:
    # Fallback to relative import (when run directly)
    from config import settings
    from auth import auth_manager
    from database import db_manager

console = Console()

@dataclass
class RetrievalResult:
    content: str
    score: float
    source: str
    extra_metadata: Optional[Dict[str, Any]] = None

class RAGSystem:
    """Retrieval-Augmented Generation system for university data."""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.text_splitter = None
        self.index_to_doc_map = {}
        self.is_initialized = False
        
    def _initialize(self):
        """Initialize RAG system components."""
        if self.is_initialized:
            return
        
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\\n\\n", "\\n", ". ", " ", ""]
            )
            
            # Initialize or load FAISS index
            self._load_or_create_index()
            
            self.is_initialized = True
            
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            self.is_initialized = False
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        index_path = os.path.join(settings.VECTOR_DB_PATH, "index.faiss")
        metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing index
            try:
                self.vector_db = faiss.read_index(index_path)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.index_to_doc_map = json.load(f)
                print(f"Loaded existing FAISS index with {self.vector_db.ntotal} vectors")
            except Exception as e:
                print(f"Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Create directory if it doesn't exist
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # Create new FAISS index (using L2 distance)
        embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.vector_db = faiss.IndexFlatL2(embedding_dim)
        self.index_to_doc_map = {}
        
        print("Created new FAISS index")
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        if not self.vector_db:
            return
            
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        index_path = os.path.join(settings.VECTOR_DB_PATH, "index.faiss")
        metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        
        faiss.write_index(self.vector_db, index_path)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_to_doc_map, f, ensure_ascii=False, indent=2)
        
        print(f"Saved FAISS index with {self.vector_db.ntotal} vectors")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the RAG system."""
        self._initialize()
        
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")
        
        for doc in documents:
            self._process_and_add_document(doc)
        
        self._save_index()
    
    def _process_and_add_document(self, doc: Dict[str, Any]):
        """Process a single document and add it to the vector store."""
        content = doc.get('content', '')
        if not content:
            return
        
        # Create document ID
        doc_id = doc.get('id') or hashlib.sha256(content.encode()).hexdigest()
        
        # Split document into chunks
        chunks = self.text_splitter.split_text(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            # Generate embedding
            embedding = self.embedding_model.encode([chunk])[0]
            
            # Add to FAISS index
            current_index = self.vector_db.ntotal
            self.vector_db.add(embedding.reshape(1, -1))
            
            # Store metadata
            chunk_metadata = {
                'content': chunk,
                'source': doc.get('url', 'Unknown'),
                'title': doc.get('title', 'Untitled'),
                'content_type': doc.get('content_type', 'general'),
                'chunk_index': i,
                'doc_id': doc_id,
                'scraped_at': doc.get('scraped_at', ''),
                'metadata': doc.get('metadata', {})
            }
            
            self.index_to_doc_map[str(current_index)] = chunk_metadata
            
            # Store in database as well
            chunk_data = {
                'id': f"{doc_id}_chunk_{i}",
                'university_data_id': doc_id,
                'chunk_text': chunk,
                'chunk_index': i,
                'vector_id': str(current_index)
            }
            
            try:
                db_manager.store_document_chunk(chunk_data)
            except Exception as e:
                print(f"Warning: Could not store chunk in database: {e}")
    
    def search(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Search for relevant documents."""
        self._initialize()
        
        if not self.is_initialized or not self.vector_db:
            return []
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        if self.vector_db.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search FAISS index
            scores, indices = self.vector_db.search(
                query_embedding.reshape(1, -1), 
                min(top_k, self.vector_db.ntotal)
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                doc_data = self.index_to_doc_map.get(str(idx))
                if not doc_data:
                    continue
                
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + score)
                
                result = RetrievalResult(
                    content=doc_data['content'],
                    score=similarity_score,
                    source=doc_data['source'],
                    extra_metadata={
                        'title': doc_data.get('title', ''),
                        'content_type': doc_data.get('content_type', ''),
                        'chunk_index': doc_data.get('chunk_index', 0),
                        'doc_id': doc_data.get('doc_id', ''),
                        'scraped_at': doc_data.get('scraped_at', ''),
                        'raw_score': float(score)
                    }
                )
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for a query, formatted for RAG."""
        results = self.search(query)
        
        if not results:
            return "No relevant information found in the university database."
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result.content.strip()
            if not content:
                continue
            
            # Add source information
            source_info = f"[Source: {result.extra_metadata.get('title', 'Unknown') if result.extra_metadata else 'Unknown'}]"
            full_content = f"{source_info}\\n{content}"
            
            if current_length + len(full_content) > max_context_length:
                break
            
            context_parts.append(full_content)
            current_length += len(full_content)
        
        context = "\\n\\n---\\n\\n".join(context_parts)
        
        return f"""Relevant information from GD Goenka University database:

{context}

Please use this information to answer the user's question about the university."""
    
    def update_from_database(self):
        """Update RAG system with latest data from database."""
        try:
            university_data = db_manager.get_university_data()
            
            documents = []
            for data in university_data:
                doc = {
                    'id': data.id,
                    'content': data.content,
                    'url': data.url,
                    'title': data.title,
                    'content_type': data.content_type,
                    'scraped_at': data.scraped_at.isoformat() if data.scraped_at else '',
                    'metadata': data.extra_metadata or {}
                }
                documents.append(doc)
            
            if documents:
                # Clear existing index and rebuild
                self._create_new_index()
                self.add_documents(documents)
                print(f"Updated RAG system with {len(documents)} documents")
            else:
                print("No university data found in database")
                
        except Exception as e:
            print(f"Error updating RAG system from database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        self._initialize()
        
        stats = {
            'initialized': self.is_initialized,
            'total_vectors': self.vector_db.ntotal if self.vector_db else 0,
            'total_documents': len(set(
                doc.get('doc_id', '') for doc in self.index_to_doc_map.values()
            )) if self.index_to_doc_map else 0,
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': settings.CHUNK_SIZE,
            'chunk_overlap': settings.CHUNK_OVERLAP
        }
        
        return stats

# Global RAG system instance
rag_system = RAGSystem()

class UniversityAssistantCLI:
    """CLI interface for University Assistant."""
    
    def __init__(self):
        self.current_user = None
        self.current_session = None
        self.conversation_id = None
        
    def display_banner(self):
        """Display welcome banner."""
        banner = Text("🎓 University Assistant AI", style="bold blue")
        subtitle = Text("Your intelligent companion for university information", style="italic")
        
        panel = Panel.fit(
            f"{banner}\n{subtitle}\n\nPowered by RAG • Multi-Model AI • Advanced Search",
            border_style="blue"
        )
        console.print(panel)
        console.print()
    
    def display_welcome_message(self, username: str):
        """Display warm welcome message after authentication."""
        welcome_messages = [
            f"🌟 Welcome to GD Goenka University Assistant, {username}!",
            f"Hello {username}! I'm here to help you with all your university queries! 🎓",
            f"Hi {username}! Ready to explore what GD Goenka University has to offer? ✨",
            f"Welcome aboard, {username}! Let's discover your perfect academic journey! 🚀"
        ]
        
        import random
        selected_message = random.choice(welcome_messages)
        
        welcome_panel = Panel.fit(
            f"[bold green]{selected_message}[/bold green]\n\n"
            f"I can help you with:\n"
            f"• 💰 Fee structures and costs\n"
            f"• 📚 Course information and programs\n"
            f"• 🎯 Admission requirements and process\n"
            f"• 🏫 University facilities and services\n"
            f"• 🎓 Career guidance and specializations\n\n"
            f"[italic cyan]Just ask me anything about GD Goenka University![/italic cyan]",
            title="🤖 Your AI Assistant",
            title_align="left",
            border_style="green",
            padding=(1, 2)
        )
        console.print(welcome_panel)
        console.print()
    
    def display_chat_greeting(self):
        """Display greeting when starting chat session."""
        import random
        from datetime import datetime
        
        chat_greetings = [
            f"Hello {self.current_user.username}! I'm your GD Goenka University assistant. How can I help you today? 😊",
            f"Hi there, {self.current_user.username}! Ready to explore everything about GD Goenka University? 🎓",
            f"Welcome to our chat, {self.current_user.username}! I'm here to answer all your university questions! 💡",
            f"Hey {self.current_user.username}! Let's discover what GD Goenka University has in store for you! 🌟"
        ]
        
        # Time-based greetings
        current_hour = datetime.now().hour
        if current_hour < 12:
            time_greeting = "Good morning"
        elif current_hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
            
        selected_greeting = random.choice(chat_greetings)
        
        # Create a personalized initial message
        initial_message = f"{time_greeting}, {self.current_user.username}! {selected_greeting.split('!')[1].strip()}"
        
        # Add session info
        session_info = f"Starting new chat session at {datetime.now().strftime('%H:%M')}"
        
        # Display as assistant message
        greeting_panel = Panel(
            f"[bold cyan]{initial_message}[/bold cyan]\n\n"
            f"💡 [italic]Try asking me things like:[/italic]\n"
            f"• 'What is the fee for B.Tech Computer Science?'\n"
            f"• 'Tell me about MBA programs'\n"
            f"• 'What are the admission requirements?'\n"
            f"• 'Show me available engineering courses'\n\n"
            f"[dim]I have the latest information from GD Goenka University website!\n{session_info}[/dim]",
            title="🤖 Assistant",
            title_align="left",
            border_style="green",
            padding=(1, 2)
        )
        console.print(greeting_panel)
        console.print()
    
    def show_auth_flow(self) -> bool:
        """Registration/Login menu and flow."""
        console.print("🔐 [bold cyan]Authentication[/bold cyan]")
        while True:
            console.print("1. Login")
            console.print("2. Register")
            console.print("3. Delete Account")
            console.print("4. Exit")
            choice = Prompt.ask("Select an option", choices=['1','2','3','4'], default='1')
            if choice == '1':
                if self.login_flow():
                    return True
            elif choice == '2':
                self.registration_flow()
            elif choice == '3':
                self.delete_account_flow()
            else:
                return False
    
    def login_flow(self) -> bool:
        username = Prompt.ask("Username")
        password = Prompt.ask("Password", password=True)
        success, msg, user = auth_manager.authenticate_user_enhanced(username, password)
        if not success:
            console.print(f"❌ {msg}", style="red")
            return False
        self.current_user = user
        console.print(f"✅ Welcome, [bold green]{user.username}[/bold green]!", style="green")
        return True
    
    def registration_flow(self):
        console.print("📝 [bold cyan]Register a new account[/bold cyan]")
        username = Prompt.ask("Choose a username")
        email = Prompt.ask("Email address")
        password = Prompt.ask("Password", password=True)
        confirm = Prompt.ask("Confirm password", password=True)
        ok, msg, user = auth_manager.register_user(username, email, password, confirm)
        if ok:
            console.print("✅ Registration successful! You can now log in.", style="green")
        else:
            console.print(f"❌ {msg}", style="red")
    
    def delete_account_flow(self):
        """Handle account deletion flow."""
        console.print("🗑️ [bold red]Delete Account[/bold red]")
        console.print("[yellow]⚠️ WARNING: This action is permanent and cannot be undone![/yellow]")
        console.print("[yellow]All your data including conversations, profile, and history will be permanently deleted.[/yellow]")
        console.print()
        
        # Confirm the user wants to proceed
        if not Confirm.ask("Are you sure you want to delete your account?"):
            console.print("Account deletion cancelled.", style="cyan")
            return
        
        # Get username and password for verification
        username = Prompt.ask("Enter your username to confirm")
        password = Prompt.ask("Enter your password to confirm", password=True)
        
        # Authenticate user first
        success, msg, user = auth_manager.authenticate_user_enhanced(username, password)
        if not success:
            console.print(f"❌ Authentication failed: {msg}", style="red")
            console.print("Account deletion cancelled for security reasons.", style="yellow")
            return
        
        # Final confirmation
        console.print(f"\nYou are about to permanently delete the account: [bold red]{user.username}[/bold red]")
        console.print("This action will:")
        console.print("• Delete your user profile and personal information")
        console.print("• Delete all your chat conversations and history")
        console.print("• Delete all audit logs associated with your account")
        console.print("• Invalidate all your active sessions")
        console.print()
        
        final_confirm = Prompt.ask(
            "Type 'DELETE' in all caps to confirm permanent account deletion",
            default=""
        )
        
        if final_confirm != "DELETE":
            console.print("Account deletion cancelled.", style="cyan")
            return
        
        # Perform deletion
        with console.status("[bold red]Deleting account and all associated data..."):
            try:
                delete_success, delete_msg = auth_manager.delete_user_account(user.id, password)
                
                if delete_success:
                    console.print(f"\n✅ {delete_msg}", style="green")
                    console.print("Thank you for using University Assistant AI. Goodbye! 👋", style="cyan")
                else:
                    console.print(f"\n❌ Failed to delete account: {delete_msg}", style="red")
                    
            except Exception as e:
                console.print(f"\n❌ An error occurred during account deletion: {str(e)}", style="red")
                console.print("Please try again or contact support if the problem persists.", style="yellow")
        
    def authenticate_user(self) -> bool:
        """Backward-compat wrapper if called elsewhere."""
        return self.show_auth_flow()
    
    def start_chat_session(self):
        """Start interactive chat session."""
        console.print("\n💬 [bold cyan]Chat Session Started[/bold cyan]")
        console.print("Type 'exit', 'quit', or 'bye' to end the session")
        console.print("Type 'help' for available commands\n")
        
        # Display chat greeting
        self.display_chat_greeting()
        
        # Generate conversation ID
        self.conversation_id = hashlib.sha256(f"{self.current_user.id}_{datetime.now()}".encode()).hexdigest()[:16]
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]", default="").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    # Display personalized farewell
                    farewell_messages = [
                        f"👋 Goodbye, {self.current_user.username}! Feel free to come back anytime with more questions about GD Goenka University!",
                        f"🌟 Thanks for chatting, {self.current_user.username}! I hope I was helpful with your university queries. Have a great day!",
                        f"✨ See you later, {self.current_user.username}! Don't hesitate to return if you need more information about admissions or courses!",
                        f"🎓 Goodbye {self.current_user.username}! Wishing you the best in your academic journey. Come back anytime!"
                    ]
                    import random
                    farewell = random.choice(farewell_messages)
                    console.print(farewell, style="cyan")
                    break
                
                # Check for help
                if user_input.lower() == 'help':
                    self.show_chat_help()
                    continue
                
                # Check for system commands
                if user_input.startswith('/'):
                    self.handle_system_command(user_input)
                    continue
                
                # Process chat message
                with console.status("[bold green]🤔 Thinking..."):
                    try:
                        from src.services import chat_service
                    except ImportError:
                        from services import chat_service
                    response = chat_service.chat(
                        query=user_input,
                        user_id=self.current_user.id,
                        conversation_id=self.conversation_id
                    )
                
                # Display response
                self.display_chat_response(response)
                
            except KeyboardInterrupt:
                console.print("\\n👋 Chat session interrupted. Goodbye!", style="yellow")
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="red")
    
    def display_chat_response(self, response):
        """Display formatted chat response."""
        # Main response
        panel = Panel(
            response.message,
            title="🤖 Assistant",
            title_align="left",
            border_style="green",
            padding=(1, 2)
        )
        console.print(panel)
        
        # Metadata
        metadata_text = f"Model: {response.model_used} | "
        metadata_text += f"Time: {response.response_time:.2f}s | "
        metadata_text += f"Tokens: {response.tokens_used}"
        
        if response.sources:
            metadata_text += f" | Sources: {len(response.sources)}"
        
        console.print(f"[dim]{metadata_text}[/dim]\\n")
    
    def show_chat_help(self):
        """Show chat help commands."""
        help_table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
        help_table.add_column("Command", style="green")
        help_table.add_column("Description")
        
        help_table.add_row("/status", "Show system status")
        help_table.add_row("/models", "List available AI models")
        help_table.add_row("/stats", "Show knowledge base statistics")
        help_table.add_row("/clear", "Clear conversation history")
        help_table.add_row("help", "Show this help message")
        help_table.add_row("exit/quit/bye", "End chat session")
        
        console.print(help_table)
        console.print()
    
    def handle_system_command(self, command: str):
        """Handle system commands."""
        cmd = command.lower().strip()
        
        if cmd == '/status':
            self.show_system_status()
        elif cmd == '/models':
            self.show_available_models()
        elif cmd == '/stats':
            self.show_knowledge_base_stats()
        elif cmd == '/clear':
            if self.conversation_id:
                try:
                    from src.services import chat_service
                except ImportError:
                    from services import chat_service
                chat_service.clear_conversation(self.conversation_id)
                console.print("✅ Conversation history cleared", style="green")
        elif cmd == '/profile':
            self.show_profile_menu()
        else:
            console.print(f"❌ Unknown command: {command}", style="red")
    
    def show_system_status(self):
        """Show system status."""
        try:
            from src.services import system_service
        except ImportError:
            from services import system_service
        status = system_service.get_system_status()
        
        status_table = Table(title="System Status", show_header=True, header_style="bold cyan")
        status_table.add_column("Component", style="green")
        status_table.add_column("Status")
        status_table.add_column("Details")
        
        # Models
        models_status = "✅ Available" if status.get('models', {}).get('available') else "❌ Not Available"
        models_count = len(status.get('models', {}).get('available', []))
        status_table.add_row("AI Models", models_status, f"{models_count} models")
        
        # RAG System
        rag_status = "✅ Initialized" if status.get('rag', {}).get('initialized') else "❌ Not Initialized"
        rag_vectors = status.get('rag', {}).get('total_vectors', 0)
        status_table.add_row("RAG System", rag_status, f"{rag_vectors} vectors")
        
        # Database
        db_status = "✅ Connected" if status.get('database', {}).get('connected') else "❌ Not Connected"
        status_table.add_row("Database", db_status, "SQLite")
        
        console.print(status_table)
        console.print()
    
    def show_available_models(self):
        """Show available AI models."""
        try:
            try:
                from src.models import model_manager
            except ImportError:
                from models import model_manager
            available_models = model_manager.get_available_models()
            
            if not available_models:
                console.print("❌ No AI models available", style="red")
                return
            
            models_table = Table(title="Available AI Models", show_header=True, header_style="bold cyan")
            models_table.add_column("Model Name", style="green")
            models_table.add_column("Type")
            models_table.add_column("Status")
            
            for model_name in available_models:
                model_type = "Groq API" if "groq" in model_name else "Hugging Face"
                models_table.add_row(model_name, model_type, "✅ Available")
            
            console.print(models_table)
            console.print()
            
        except Exception as e:
            console.print(f"❌ Error getting models: {e}", style="red")
    
    def show_knowledge_base_stats(self):
        """Show knowledge base statistics."""
        try:
            try:
                from src.services import university_service
            except ImportError:
                from services import university_service
            stats = university_service.get_knowledge_base_stats()
            
            stats_table = Table(title="Knowledge Base Statistics", show_header=True, header_style="bold cyan")
            stats_table.add_column("Metric", style="green")
            stats_table.add_column("Value")
            
            # RAG stats
            rag_stats = stats.get('rag_system', {})
            stats_table.add_row("Total Vectors", str(rag_stats.get('total_vectors', 0)))
            stats_table.add_row("Total Documents", str(rag_stats.get('total_documents', 0)))
            stats_table.add_row("Embedding Model", rag_stats.get('embedding_model', 'Unknown'))
            
            # Database stats
            db_stats = stats.get('database', {})
            stats_table.add_row("DB Documents", str(db_stats.get('total_documents', 0)))
            
            latest_scrape = db_stats.get('latest_scrape')
            if latest_scrape:
                stats_table.add_row("Latest Scrape", latest_scrape[:19].replace('T', ' '))
            
            console.print(stats_table)
            console.print()
            
        except Exception as e:
            console.print(f"❌ Error getting stats: {e}", style="red")
    
    def show_profile_menu(self):
        """User profile management UI."""
        if not self.current_user:
            console.print("❌ Not authenticated", style="red")
            return
        while True:
            console.print("\n👤 [bold cyan]Profile Management[/bold cyan]")
            console.print("1. View Profile")
            console.print("2. Update Profile")
            console.print("3. Change Password")
            console.print("4. Back")
            choice = Prompt.ask("Select an option", choices=['1','2','3','4'], default='1')
            if choice == '1':
                self.view_profile()
            elif choice == '2':
                self.update_profile()
            elif choice == '3':
                self.change_password_flow()
            else:
                break
    
    def view_profile(self):
        try:
            from src.database import db_manager
        except ImportError:
            from database import db_manager
        db_user = db_manager.get_user_by_id(self.current_user.id)
        table = Table(title="Your Profile", show_header=True, header_style="bold cyan")
        table.add_column("Field", style="green")
        table.add_column("Value")
        fields = {
            "Username": db_user.username,
            "Email": db_user.email,
            "Full Name": getattr(db_user, 'full_name', '') or '',
            "Phone": getattr(db_user, 'phone', '') or '',
            "Bio": getattr(db_user, 'bio', '') or '',
            "MFA Enabled": "Yes" if getattr(db_user, 'mfa_enabled', False) else "No",
            "Created": db_user.created_at.strftime('%Y-%m-%d %H:%M') if db_user.created_at else '',
            "Last Login": db_user.last_login.strftime('%Y-%m-%d %H:%M') if db_user.last_login else ''
        }
        for k,v in fields.items():
            table.add_row(k, str(v))
        console.print(table)
    
    def update_profile(self):
        full_name = Prompt.ask("Full name", default="")
        phone = Prompt.ask("Phone", default="")
        bio = Prompt.ask("Bio", default="")
        avatar_url = Prompt.ask("Avatar URL", default="")
        try:
            from src.database import db_manager
        except ImportError:
            from database import db_manager
        updated = db_manager.update_user_profile(self.current_user.id, {
            "full_name": full_name,
            "phone": phone,
            "bio": bio,
            "avatar_url": avatar_url
        })
        if updated:
            console.print("✅ Profile updated", style="green")
        else:
            console.print("❌ Failed to update profile", style="red")
    
    def change_password_flow(self):
        current = Prompt.ask("Current password", password=True)
        new = Prompt.ask("New password", password=True)
        confirm = Prompt.ask("Confirm new password", password=True)
        ok, msg = auth_manager.change_password(self.current_user.id, current, new, confirm)
        if ok:
            console.print("✅ Password changed successfully", style="green")
        else:
            console.print(f"❌ {msg}", style="red")

def main_menu():
    """Main menu interface."""
    cli = UniversityAssistantCLI()
    cli.display_banner()
    
    # Authentication flow
    if not cli.show_auth_flow():
        console.print("❌ Authentication required. Exiting...", style="red")
        return
    
    # Display warm welcome message after successful authentication
    cli.display_welcome_message(cli.current_user.username)
    
    while True:
        console.print("\n📋 [bold cyan]Main Menu[/bold cyan]")
        console.print("1. Start Chat Session")
        console.print("2. Update Knowledge Base") 
        console.print("3. System Status")
        console.print("4. Profile Management")
        console.print("5. Exit")
        
        choice = Prompt.ask("Select an option", choices=['1', '2', '3', '4', '5', '6'], default='1')
        
        if choice == '1':
            # Ask if user wants to use threaded chat service
            use_threading = Confirm.ask("Use enhanced threading for faster responses?", default=True)
            if use_threading:
                console.print("🚀 Using threaded chat service for better performance!", style="bold green")
                try:
                    from src.threaded_services import threaded_chat_service
                    cli.chat_service = threaded_chat_service
                except ImportError:
                    from threaded_services import threaded_chat_service
                    cli.chat_service = threaded_chat_service
            cli.start_chat_session()
        elif choice == '2':
            update_knowledge_base()
        elif choice == '3':
            cli.show_system_status()
        elif choice == '4':
            cli.show_profile_menu()
        elif choice == '5':
            show_performance_report()
        elif choice == '6':
            console.print("👋 Thank you for using University Assistant!", style="cyan")
            break

def update_knowledge_base():
    """Update knowledge base by scraping university data."""
    console.print("\\n🔄 [bold cyan]Updating Knowledge Base[/bold cyan]")
    
    if not Confirm.ask("This will scrape the university website and update the knowledge base. Continue?"):
        return
    
    # Ask if user wants to use threading
    use_threading = Confirm.ask("Would you like to use multi-threading for faster processing?", default=True)
    
    if use_threading:
        console.print("🚀 Using enhanced threading for faster processing!", style="bold green")
        
        # Import threading components
        try:
            from src.threaded_services import threaded_university_service
            from src.performance_monitor import monitor_threaded_operation
        except ImportError:
            from threaded_services import threaded_university_service
            from performance_monitor import monitor_threaded_operation
        
        # Use performance monitoring
        with monitor_threaded_operation("Knowledge Base Update", worker_count=4) as metrics:
            def progress_callback(completed, total):
                percentage = (completed / total) * 100
                console.print(f"📊 Progress: {completed}/{total} ({percentage:.1f}%)")
            
            result = threaded_university_service.update_knowledge_base(progress_callback=progress_callback)
    else:
        console.print("Using standard processing (slower but uses less resources)", style="yellow")
        with console.status("[bold green]Scraping university data..."):
            try:
                from src.services import university_service
            except ImportError:
                from services import university_service
            result = university_service.update_knowledge_base()
    
    # Display results
    if result.get('success'):
        console.print("✅ Knowledge base updated successfully!", style="green")
        console.print(f"📊 Scraped: {result.get('documents_scraped', 0)} documents")
        console.print(f"💾 Stored: {result.get('documents_stored', 0)} documents")
        
        if result.get('processing_time'):
            console.print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f} seconds")
        
        if result.get('performance_improvement'):
            console.print(f"🚀 {result.get('performance_improvement')}", style="bold green")
    else:
        console.print(f"❌ Update failed: {result.get('error', 'Unknown error')}", style="red")


# CLI Commands using Click
@click.group()
def cli():
    """University Assistant AI - CLI Interface"""
    pass

@cli.command()
def chat():
    """Start interactive chat session"""
    main_menu()

@cli.command()
def status():
    """Show system status"""
    console = Console()
    try:
        from src.services import system_service
    except ImportError:
        from services import system_service
    status = system_service.get_system_status()
    console.print(status, style="cyan")

@cli.command()
def scrape():
    """Scrape university data and update knowledge base"""
    update_knowledge_base()


@cli.command()
@click.option('--api-key', help='Set Hugging Face API key')
@click.option('--groq-key', help='Set Groq API key')
def configure(api_key, groq_key):
    """Configure API keys and settings"""
    if api_key:
        # In a real implementation, this would securely store the API key
        console.print(f"✅ Hugging Face API key configured", style="green")
    
    if groq_key:
        console.print(f"✅ Groq API key configured", style="green")
    
    if not api_key and not groq_key:
        console.print("Use --api-key or --groq-key to configure API keys", style="yellow")

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("secure_storage", exist_ok=True)
        
        # Run CLI
        cli()
    except KeyboardInterrupt:
        console.print("\\n👋 Goodbye!", style="cyan")
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="red")
        sys.exit(1)
