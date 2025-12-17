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
                if (settings.HUGGINGFACE_API_TOKEN and 
                    settings.HUGGINGFACE_API_TOKEN.strip() and 
                    settings.HUGGINGFACE_API_TOKEN != "your_huggingface_token_here"):
                    
                    self._use_api = True
                    self._api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                    self._headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_TOKEN}"}
                    print(f"✅ [Threading] Using Hugging Face API for {self.model_name}")
                else:
                    # Fallback to local model
                    self._use_api = False
                    print(f"🔄 [Threading] Loading local model: {self.model_name}")
                    
                    # Initialize base components
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
                print(f"❌ [Threading] Error initializing Hugging Face model {self.model_name}: {e}")
                self._initialized = False
    
    def _get_pipeline(self):
        """Get a pipeline instance from the queue."""
        if self._use_api:
            return None
        return self.model_queue.get()
    
    def _return_pipeline(self, pipe):
        """Return a pipeline instance to the queue."""
        if pipe and not self._use_api:
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
        
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE
        
        if self._use_api:
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
                
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
                print(f"⚠️ [Threading] GROQ_API_KEY not provided for {self.model_name}")
                return
                
            try:
                print(f"🚀 [Threading] Initializing Groq client for: {self.model_name}")
                self.client = Groq(api_key=settings.GROQ_API_KEY)
                self._initialized = True
                print(f"✅ [Threading] Successfully initialized Groq client for {self.model_name}")
            except Exception as e:
                print(f"❌ [Threading] Error initializing Groq client for {self.model_name}: {e}")
                self._initialized = False
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> ModelResponse:
        """Generate text using Groq API with threading support."""
        self._initialize()
        
        if not self._initialized:
            raise RuntimeError(f"Groq client for {self.model_name} not initialized")
        
        start_time = time.time()
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE
        
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
            return self._initialized and bool(settings.GROQ_API_KEY)
        except:
            return False

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
                print(f"❌ Failed to initialize {name}: {e}")
        
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
                print(f"❌ Failed to initialize {name}: {e}")
        
        print(f"🎉 Threaded model manager initialized with {len(self.models)} models")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        available = []
        for name, model in self.models.items():
            if model.is_available():
                available.append(name)
        return available
    
    def generate(self, prompt: str, model_name: Optional[str] = None, 
                max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> ModelResponse:
        """Generate response using specified model or default."""
        model_name = model_name or self.get_best_available_model()
        
        if not model_name:
            raise RuntimeError("No available models")
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        if not model.is_available():
            raise RuntimeError(f"Model '{model_name}' is not available")
        
        return model.generate(prompt, max_tokens, temperature)
    
    async def generate_async(self, prompt: str, model_name: Optional[str] = None,
                           max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> ModelResponse:
        """Generate response asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, model_name, max_tokens, temperature)
    
    def generate_concurrent(self, prompts: List[str], model_name: Optional[str] = None,
                          max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> List[ModelResponse]:
        """Generate responses for multiple prompts concurrently."""
        model_name = model_name or self.get_best_available_model()
        
        if not model_name:
            raise RuntimeError("No available models")
        
        print(f"🚀 Generating {len(prompts)} responses concurrently with {model_name}...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ConcurrentGen") as executor:
            futures = {
                executor.submit(self.generate, prompt, model_name, max_tokens, temperature): i 
                for i, prompt in enumerate(prompts)
            }
            
            responses = [None] * len(prompts)  # Maintain order
            
            for future in as_completed(futures):
                index = futures[future]
                try:
                    response = future.result(timeout=120)
                    responses[index] = response
                except Exception as e:
                    error_response = ModelResponse(
                        content=f"Error generating response: {str(e)}",
                        model=model_name or "unknown",
                        tokens_used=0,
                        response_time=0,
                        extra_metadata={"error": str(e)}
                    )
                    responses[index] = error_response
        
        elapsed_time = time.time() - start_time
        print(f"🎉 Concurrent generation completed in {elapsed_time:.2f} seconds!")
        
        return responses
    
    def get_best_available_model(self) -> Optional[str]:
        """Get the best available model based on priority."""
        # Priority order: Groq models (faster) -> Hugging Face models
        priority_models = [
            "groq-llama", "groq-llama-70b", "groq-gemma", 
            "microsoft-dialogpt", "gpt2", "distilgpt2"
        ]
        
        available_models = self.get_available_models()
        
        for model_name in priority_models:
            if model_name in available_models:
                return model_name
        
        # Return any available model
        return available_models[0] if available_models else None
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about all models."""
        stats = {
            "total_models": len(self.models),
            "available_models": self.get_available_models(),
            "threading_enabled": True,
            "max_workers": self.max_workers,
            "models_detail": {}
        }
        
        for name, model in self.models.items():
            stats["models_detail"][name] = {
                "available": model.is_available(),
                "type": type(model).__name__,
                "thread_pool_size": getattr(model, 'max_workers', 1)
            }
        
        return stats
    
    def cleanup(self):
        """Clean up all model resources."""
        print("🧹 Cleaning up threaded models...")
        for name, model in self.models.items():
            try:
                model.cleanup()
            except Exception as e:
                print(f"⚠️ Error cleaning up {name}: {e}")
        print("✅ Model cleanup completed")

# Global threaded model manager instance
threaded_model_manager = ThreadedModelManager(max_workers=4)