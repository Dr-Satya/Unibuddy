import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.config import settings
from src.models import model_manager, ModelResponse
# RAG system import will be handled as a forward reference
# from src.main import rag_system, RetrievalResult
from src.database import db_manager
from src.auth import auth_manager
from src.scraper import university_scraper

@dataclass
class ChatMessage:
    content: str
    role: str  # 'user', 'assistant', 'system'
    timestamp: Optional[datetime] = None
    extra_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ChatResponse:
    message: str
    model_used: str
    response_time: float
    sources: Optional[List[str]] = None
    tokens_used: int = 0
    extra_metadata: Optional[Dict[str, Any]] = None

class ChatService:
    """Main chat service that orchestrates RAG, models, and conversation management."""
    
    def __init__(self):
        self.conversation_history: Dict[str, List[ChatMessage]] = {}
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the university assistant."""
        return """You are a helpful AI assistant for GD Goenka University. Your role is to:

1. Answer questions about university admissions, fees, courses, facilities, and general information
2. Provide accurate and helpful information based on the university's official data
3. Be polite, professional, and student-friendly
4. If you don't have specific information, acknowledge this and suggest contacting the university directly
5. Use the provided context from the university database to answer questions accurately

Guidelines:
- Always prioritize accuracy over assumptions
- Cite sources when providing specific information like fees or admission requirements
- Be helpful and encouraging to prospective students
- If asked about personal or sensitive information, redirect appropriately
- Keep responses concise but comprehensive

You have access to the latest information from the GD Goenka University website and database."""
    
    def _build_conversation_context(self, conversation_id: str, max_messages: int = 10) -> str:
        """Build conversation context from recent messages."""
        if conversation_id not in self.conversation_history:
            return ""
        
        messages = self.conversation_history[conversation_id][-max_messages:]
        
        context_parts = []
        for msg in messages:
            role_prefix = "Human: " if msg.role == "user" else "Assistant: "
            context_parts.append(f"{role_prefix}{msg.content}")
        
        return "\\n\\n".join(context_parts)
    
    def _format_prompt_with_rag(self, query: str, context: str, conversation_context: str = "") -> str:
        """Format the final prompt with RAG context and conversation history."""
        system_prompt = self._create_system_prompt()
        
        prompt_parts = [system_prompt]
        
        if context and "No relevant information found" not in context:
            prompt_parts.append(f"\\n\\nRelevant Information:\\n{context}")
        
        if conversation_context:
            prompt_parts.append(f"\\n\\nConversation History:\\n{conversation_context}")
        
        prompt_parts.append(f"\\n\\nUser Question: {query}")
        prompt_parts.append("\\n\\nAssistant:")
        
        return "\\n".join(prompt_parts)
    
    async def chat_async(self, query: str, user_id: Optional[str] = None, 
                        conversation_id: Optional[str] = None,
                        model_name: Optional[str] = None) -> ChatResponse:
        """Process chat message asynchronously with RAG enhancement."""
        start_time = time.time()
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = hashlib.sha256(f"{user_id}_{time.time()}".encode()).hexdigest()[:16]
        
        # Initialize conversation history if needed
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        try:
            # Get RAG context (import at runtime to avoid circular import)
            from src.main import rag_system
            rag_context = rag_system.get_context_for_query(query, max_context_length=1500)
            
            # Build conversation context
            conversation_context = self._build_conversation_context(conversation_id)
            
            # Format the final prompt
            formatted_prompt = self._format_prompt_with_rag(query, rag_context, conversation_context)
            
            # Generate response using AI model
            model_response = await model_manager.generate_async(
                formatted_prompt,
                model_name=model_name
            )
            
            response_time = time.time() - start_time
            
            # Extract sources from RAG results
            rag_results = rag_system.search(query)
            sources = [result.source for result in rag_results[:3]]  # Top 3 sources
            
            # Add messages to conversation history
            user_message = ChatMessage(content=query, role="user")
            assistant_message = ChatMessage(
                content=model_response.content,
                role="assistant",
                extra_metadata={
                    "model": model_response.model,
                    "tokens": model_response.tokens_used,
                    "sources": sources
                }
            )
            
            self.conversation_history[conversation_id].extend([user_message, assistant_message])
            
            # Keep conversation history manageable
            if len(self.conversation_history[conversation_id]) > settings.MAX_CONVERSATION_LENGTH:
                self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-settings.MAX_CONVERSATION_LENGTH:]
            
            # Store in database if user_id provided
            if user_id:
                try:
                    # Store user message
                    user_msg_data = {
                        "id": hashlib.sha256(f"{conversation_id}_{user_message.timestamp}_{query}".encode()).hexdigest(),
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "content": query,
                        "role": "user",
                        "created_at": user_message.timestamp
                    }
                    db_manager.create_message(user_msg_data)
                    
                    # Store assistant message
                    assistant_msg_data = {
                        "id": hashlib.sha256(f"{conversation_id}_{assistant_message.timestamp}_{model_response.content}".encode()).hexdigest(),
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "content": model_response.content,
                        "role": "assistant",
                        "model_used": model_response.model,
                        "tokens_used": model_response.tokens_used,
                        "response_time": response_time,
                        "created_at": assistant_message.timestamp,
                        "extra_metadata": {
                            "sources": sources,
                            "rag_context_length": len(rag_context) if rag_context else 0
                        }
                    }
                    db_manager.create_message(assistant_msg_data)
                    
                except Exception as e:
                    print(f"Warning: Could not store messages in database: {e}")
            
            return ChatResponse(
                message=model_response.content,
                model_used=model_response.model,
                response_time=response_time,
                sources=sources,
                tokens_used=model_response.tokens_used,
                extra_metadata={
                    "conversation_id": conversation_id,
                    "rag_context_length": len(rag_context) if rag_context else 0,
                    "num_sources": len(sources)
                }
            )
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error processing your request: {str(e)}"
            
            return ChatResponse(
                message=error_message,
                model_used="error",
                response_time=time.time() - start_time,
                sources=[],
                tokens_used=0,
                extra_metadata={"error": str(e)}
            )
    
    def chat(self, query: str, user_id: Optional[str] = None, 
             conversation_id: Optional[str] = None,
             model_name: Optional[str] = None) -> ChatResponse:
        """Synchronous wrapper for chat functionality."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.chat_async(query, user_id, conversation_id, model_name)
            )
        finally:
            loop.close()
    
    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history."""
        return self.conversation_history.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]

class UniversityDataService:
    """Service for managing university data and knowledge base."""
    
    def __init__(self):
        pass
    
    def update_knowledge_base(self) -> Dict[str, Any]:
        """Update the knowledge base by scraping and processing university data."""
        try:
            print("🔄 Starting knowledge base update...")
            
            # Scrape university data
            documents = university_scraper.scrape_all()
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents scraped",
                    "documents_scraped": 0,
                    "documents_stored": 0
                }
            
            # Store in database
            stored_count = university_scraper.store_scraped_data(documents)
            
            # Update RAG system
            from src.main import rag_system
            rag_system.update_from_database()
            
            print(f"✅ Knowledge base update completed")
            
            return {
                "success": True,
                "documents_scraped": len(documents),
                "documents_stored": stored_count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error updating knowledge base: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_scraped": 0,
                "documents_stored": 0
            }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            from src.main import rag_system
            rag_stats = rag_system.get_statistics()
            
            # Get database stats
            university_data = db_manager.get_university_data()
            
            content_types = {}
            for data in university_data:
                content_type = data.content_type or "unknown"
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                "rag_system": rag_stats,
                "database": {
                    "total_documents": len(university_data),
                    "content_types": content_types,
                    "latest_scrape": max([data.scraped_at for data in university_data]).isoformat() if university_data else None
                },
                "models": {
                    "available_models": model_manager.get_available_models(),
                    "default_model": settings.DEFAULT_MODEL
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def search_knowledge_base(self, query: str, top_k: int = 5):
        """Search the knowledge base."""
        from src.main import rag_system
        return rag_system.search(query, top_k)

class SystemService:
    """Service for system-level operations and health checks."""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        try:
            from src.main import rag_system
            return {
                "timestamp": datetime.now().isoformat(),
                "models": {
                    "available": model_manager.get_available_models(),
                    "total_models": len(model_manager.models),
                    "default_model": settings.DEFAULT_MODEL
                },
                "rag": {
                    "initialized": rag_system.is_initialized,
                    "total_vectors": rag_system.vector_db.ntotal if rag_system.vector_db else 0
                },
                "database": {
                    "connected": True,  # If we get here, DB is working
                    "url": settings.DATABASE_URL
                },
                "configuration": {
                    "chunk_size": settings.CHUNK_SIZE,
                    "max_tokens": settings.MAX_TOKENS,
                    "temperature": settings.TEMPERATURE,
                    "top_k_results": settings.TOP_K_RESULTS
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global service instances
chat_service = ChatService()
university_service = UniversityDataService()
system_service = SystemService()
