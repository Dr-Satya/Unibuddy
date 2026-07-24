import asyncio
import time
import threading
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

import requests
from groq import Groq

from src.config import settings

@dataclass
class ModelResponse:
    content: str
    model: str
    tokens_used: int
    response_time: float
    extra_metadata: Optional[Dict[str, Any]] = None

class BaseThreadedModel(ABC):
    """Abstract base class for all threaded AI models."""
    
    def __init__(self):
        self.initialization_lock = threading.Lock()
        self._initialized = False
        self.thread_pool = None
        
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None) -> ModelResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    def cleanup(self):
        """Clean up resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

class ThreadedHuggingFaceModel(BaseThreadedModel):
    """Multi-threaded Hugging Face Transformers model integration."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", max_workers: int = 2):
        super().__init__()
        self.model_name = model_name
        self.max_workers = max_workers
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"HF-{model_name}")
        
        # Queue for managing model instances across threads
        self.model_queue = queue.Queue()
        
    def _initialize(self):
        """Lazy initialization of the model with thread safety."""
        if self._initialized:
            return
            
        with self.initialization_lock:
            if self._initialized:  # Double-check pattern
                return
                
            try:
                print(f"🚀 [Threading] Initializing Hugging Face model: {self.model_name}")
                
                # Try to use API first (faster and more thread-friendly)
                if (getattr(settings, 'HUGGINGFACE_API_TOKEN', None) and 
                    str(getattr(settings, 'HUGGINGFACE_API_TOKEN')).strip() and 
                    str(getattr(settings, 'HUGGINGFACE_API_TOKEN')) != "your_huggingface_token_here"):
                    
                    self._use_api = True
                    self._api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                    self._headers = {"Authorization": f"Bearer {getattr(settings, 'HUGGINGFACE_API_TOKEN')}"}
                    print(f"✅ [Threading] Using Hugging Face API for {self.model_name}")
                else:
                    # Fallback to local model
                    self._use_api = False
                    print(f"🔄 [Threading] Loading local model: {self.model_name}")
                    
                    # Initialize base components
                    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, 
                        device_map="cpu",  # Force CPU for consistency
                        torch_dtype="auto",
                        low_cpu_mem_usage=True
                    )
                    
                    # Create multiple pipeline instances for threading
                    for i in range(self.max_workers):
                        pipe = pipeline(
                            "text-generation",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device="cpu"  # Force CPU
                        )
                        self.model_queue.put(pipe)
                    
                    print(f"✅ [Threading] Created {self.max_workers} pipeline instances for {self.model_name}")
                
                self._initialized = True
                print(f"🎉 [Threading] Successfully initialized {self.model_name}")
                
            except Exception as e:
                # Log as warning to avoid alarming icon; still record exception text
                print(f"⚠️ [Threading] Warning initializing Hugging Face model {self.model_name}: {e} (skipping)")
                self._initialized = False
    
    def _get_pipeline(self):
        """Get a pipeline instance from the queue."""
        if getattr(self, '_use_api', False):
            return None
        return self.model_queue.get()
    
    def _return_pipeline(self, pipe):
        """Return a pipeline instance to the queue."""
        if pipe and not getattr(self, '_use_api', False):
            self.model_queue.put(pipe)
    
    def _generate_with_api(self, prompt: str, max_tokens: int, temperature: float) -> ModelResponse:
        """Generate text using Hugging Face API."""
        start_time = time.time()
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(
                self._api_url, 
                headers=self._headers,
                json=payload,
                timeout=45
            )
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                content = result["generated_text"]
            else:
                content = str(result)
            
            # Clean up the content
            content = content.strip()
            if not content:
                content = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
            
            # Estimate tokens (rough approximation)
            tokens_used = int(len(content.split()) * 1.3)
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                extra_metadata={"method": "api", "thread_id": threading.current_thread().name}
            )
            
        except Exception as e:
            error_content = f"API Error: {str(e)}. Please try again or use a different model."
            return ModelResponse(
                content=error_content,
                model=self.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                extra_metadata={"error": str(e), "method": "api"}
            )
    
    def _generate_with_local(self, prompt: str, max_tokens: int, temperature: float) -> ModelResponse:
        """Generate text using local model."""
        start_time = time.time()
        pipe = self._get_pipeline()
        
        try:
            print(f"🧠 [Thread {threading.current_thread().name}] Generating with local model...")
            
            # Configure generation parameters
            generation_args = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            outputs = pipe(prompt, **generation_args)
            
            if outputs and len(outputs) > 0:
                content = outputs[0]["generated_text"]
            else:
                content = "I apologize, but I couldn't generate a response."
            
            # Clean up the content
            content = content.strip()
            if not content:
                content = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
            
            tokens_used = len(self.tokenizer.encode(prompt + content))
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                extra_metadata={"method": "local", "thread_id": threading.current_thread().name}
            )
            
        except Exception as e:
            error_content = f"Local model error: {str(e)}. The model might be overloaded."
            return ModelResponse(
                content=error_content,
                model=self.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                extra_metadata={"error": str(e), "method": "local"}
            )
        finally:
            self._return_pipeline(pipe)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None) -> ModelResponse:
        """Generate text using Hugging Face model with threading support."""
        self._initialize()
        
        if not self._initialized:
            raise RuntimeError(f"Model {self.model_name} not initialized")
        
        max_tokens = max_tokens or getattr(settings, 'MAX_TOKENS', 512)
        temperature = temperature or getattr(settings, 'TEMPERATURE', 0.2)
        
        if getattr(self, '_use_api', False):
            return self._generate_with_api(prompt, max_tokens, temperature)
        else:
            return self._generate_with_local(prompt, max_tokens, temperature)
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        try:
            self._initialize()
            return self._initialized
        except:
            return False

class ThreadedGroqModel(BaseThreadedModel):
    """Multi-threaded Groq API integration."""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant", max_workers: int = 4):
        super().__init__()
        self.model_name = model_name
        self.max_workers = max_workers
        self.client = None
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"Groq-{model_name}")
        
    def _initialize(self):
        """Initialize Groq client with thread safety."""
        if self._initialized:
            return
            
        with self.initialization_lock:
            if self._initialized:  # Double-check pattern
                return
                
            if not getattr(settings, 'GROQ_API_KEY', None) or getattr(settings, 'GROQ_API_KEY') == "your_groq_api_key_here":
                print(f"⚠️ [Threading] GROQ_API_KEY not provided for {self.model_name}")
                return
                
            try:
                print(f"🚀 [Threading] Initializing Groq client for: {self.model_name}")
                self.client = Groq(api_key=getattr(settings, 'GROQ_API_KEY'))
                self._initialized = True
                print(f"✅ [Threading] Successfully initialized Groq client for {self.model_name}")
            except Exception as e:
                # warn instead of error
                print(f"⚠️ [Threading] Warning initializing Groq client for {self.model_name}: {e} (skipping)")
                self._initialized = False
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> ModelResponse:
        """Generate text using Groq API with threading support."""
        self._initialize()
        
        if not self._initialized:
            raise RuntimeError(f"Groq client for {self.model_name} not initialized")
        
        start_time = time.time()
        max_tokens = max_tokens or getattr(settings, 'MAX_TOKENS', 512)
        temperature = temperature or getattr(settings, 'TEMPERATURE', 0.2)
        
        try:
            print(f"🚀 [Thread {threading.current_thread().name}] Groq generation starting...")
            
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
            
            print(f"✅ [Thread {threading.current_thread().name}] Groq generation completed in {response_time:.2f}s")
            
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
                    },
                    "thread_id": threading.current_thread().name
                }
            )
            
        except Exception as e:
            error_content = f"Groq API Error: {str(e)}. Please try again."
            return ModelResponse(
                content=error_content,
                model=self.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                extra_metadata={"error": str(e)}
            )
    
    def is_available(self) -> bool:
        """Check if Groq API is available."""
        try:
            self._initialize()
            return self._initialized and bool(getattr(settings, 'GROQ_API_KEY', None))
        except:
            return False

    def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None):
        """FEATURE 2 (streaming): generator that yields response text chunks
        as they arrive from Groq, instead of waiting for the full
        completion. Purely additive -- generate() above is untouched, so
        any existing caller of generate() keeps working exactly as before.
        Uses the Groq SDK's native stream=True support, the standard
        OpenAI-compatible streaming interface Groq's API exposes.

        Yields plain text deltas (str). On error, yields a single error
        message string and stops, mirroring generate()'s existing
        error-as-content behavior so callers don't need new exception
        handling beyond what they'd already need for generate()."""
        self._initialize()

        if not self._initialized:
            yield f"Groq client for {self.model_name} not initialized"
            return

        max_tokens = max_tokens or getattr(settings, 'MAX_TOKENS', 512)
        temperature = temperature or getattr(settings, 'TEMPERATURE', 0.2)

        try:
            messages = [{"role": "user", "content": prompt}]
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            yield f"Groq API Error: {str(e)}. Please try again."

class ThreadedModelManager:
    """Enhanced multi-threaded model manager with concurrent processing capabilities."""
    
    def __init__(self, max_workers: int = 4):
        self.models: Dict[str, BaseThreadedModel] = {}
        self.max_workers = max_workers
        self.initialization_lock = threading.Lock()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models with threading support."""
        print("🚀 Initializing threaded models...")
        
        # Initialize Hugging Face models
        hf_models = [
            ("gpt2", "gpt2"),
            ("distilgpt2", "distilgpt2"),
            ("microsoft-dialogpt", "microsoft/DialoGPT-medium"),
        ]
        
        for name, model_path in hf_models:
            try:
                self.models[name] = ThreadedHuggingFaceModel(model_path, max_workers=2)
                print(f"✅ Initialized threaded HuggingFace model: {name}")
            except Exception as e:
                print(f"⚠️ Failed to initialize {name}: {e}")
        
        # Initialize Groq models
        groq_models = [
            ("groq-llama", "llama-3.1-8b-instant"),
            ("groq-llama-70b", "llama-3.1-70b-versatile"),
            ("groq-gemma", "gemma2-9b-it"),
        ]
        
        for name, model_name in groq_models:
            try:
                self.models[name] = ThreadedGroqModel(model_name, max_workers=4)
                print(f"✅ Initialized threaded Groq model: {name}")
            except Exception as e:
                print(f"⚠️ Failed to initialize {name}: {e}")
        
        print(f"🎉 Threaded model manager initialized with {len(self.models)} models")