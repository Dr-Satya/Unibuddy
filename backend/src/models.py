import asyncio
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from groq import Groq

from src.config import settings

@dataclass
class ModelResponse:
    content: str
    model: str
    tokens_used: int
    response_time: float
    extra_metadata: Optional[Dict[str, Any]] = None

class BaseModel(ABC):
    """Abstract base class for all AI models."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None) -> ModelResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class HuggingFaceModel(BaseModel):
    """Hugging Face Transformers model integration."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
            
        try:
            # Always use local models for reliability
            self._use_api = False
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self._initialized = True
        except Exception as e:
            print(f"Error initializing Hugging Face model: {e}")
            self._initialized = False
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None) -> ModelResponse:
        """Generate text using Hugging Face model."""
        self._initialize()
        
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE
        
        try:
            # Use local model
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            content = outputs[0]["generated_text"]
            tokens_used = len(self.tokenizer.encode(prompt + content))
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=content.strip(),
                model=self.model_name,
                tokens_used=int(tokens_used),
                response_time=response_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        try:
            self._initialize()
            return self._initialized
        except:
            return False

class GroqModel(BaseModel):
    """Groq API integration."""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = None
        self._initialized = False
        
    def _initialize(self):
        """Initialize Groq client."""
        if self._initialized:
            return
            
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not provided")
            
        try:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            self._initialized = True
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self._initialized = False
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> ModelResponse:
        """Generate text using Groq API."""
        self._initialize()
        
        if not self._initialized:
            raise RuntimeError("Groq client not initialized")
        
        start_time = time.time()
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            content = chat_completion.choices[0].message.content
            tokens_used = chat_completion.usage.total_tokens
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                extra_metadata={
                    "usage": {
                        "prompt_tokens": chat_completion.usage.prompt_tokens,
                        "completion_tokens": chat_completion.usage.completion_tokens,
                        "total_tokens": chat_completion.usage.total_tokens
                    }
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Error with Groq API: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Groq API is available."""
        try:
            self._initialize()
            return self._initialized and bool(settings.GROQ_API_KEY)
        except:
            return False

class ModelManager:
    """Manages multiple AI models and provides a unified interface."""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models."""
        # Initialize Groq models only for reliability
        groq_models = [
            ("groq-llama", "llama-3.1-8b-instant"),
            ("groq-llama-70b", "llama-3.1-70b-versatile"),
            ("groq-gemma", "gemma2-9b-it"),
            ("groq-mixtral", "mixtral-8x7b-32768"),
        ]
        
        for name, model_name in groq_models:
            try:
                self.models[name] = GroqModel(model_name)
                print(f"✅ Initialized {name} model")
            except Exception as e:
                print(f"❌ Failed to initialize {name}: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return [name for name, model in self.models.items() if model.is_available()]
    
    def generate(self, prompt: str, model_name: Optional[str] = None, 
                max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> ModelResponse:
        """Generate response using specified model or default."""
        model_name = model_name or settings.DEFAULT_MODEL
        
        # If the requested model is not available, use the best available one
        if model_name not in self.models or not self.models[model_name].is_available():
            best_model = self.get_best_available_model()
            if not best_model:
                raise RuntimeError("No models are available")
            model_name = best_model
        
        model = self.models[model_name]
        return model.generate(prompt, max_tokens, temperature)
    
    def get_best_available_model(self) -> Optional[str]:
        """Get the best available model."""
        # Priority order: Groq models only (fast and reliable)
        priority_models = ["groq-llama", "groq-llama-70b", "groq-mixtral", "groq-gemma"]
        
        available_models = self.get_available_models()
        
        for model_name in priority_models:
            if model_name in available_models:
                return model_name
        
        # Return any available model
        return available_models[0] if available_models else None
    
    async def generate_async(self, prompt: str, model_name: Optional[str] = None,
                           max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> ModelResponse:
        """Generate response asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, model_name, max_tokens, temperature)

# Global model manager instance
model_manager = ModelManager()
