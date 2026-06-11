import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Optional
import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        """Runs a standard chat completion and returns LLMResponse."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams response tokens, yielding dicts containing:
        - "token": str (the partial text chunk)
        - "metadata": dict (only yielded in the very last chunk, containing token counts and latency)
        """
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4o-mini") -> None:
        key = api_key or settings.openai_api_key
        # We allow key to be empty only if we mock or default in tests
        self.client = AsyncOpenAI(api_key=key or "dummy-key")
        self.default_model = default_model

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        model = kwargs.get("model", self.default_model)
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.time() - start_time) * 1000
            
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            text = response.choices[0].message.content or ""

            return LLMResponse(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms
            )
        except Exception as e:
            logger.error("openai_completion_failed", error=str(e), model=model)
            raise e

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        model = kwargs.get("model", self.default_model)
        start_time = time.time()

        try:
            stream_response = await self.client.chat.completions.create(
                model=model,
                messages=messages, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            prompt_tokens = 0
            completion_tokens = 0

            async for chunk in stream_response:
                # If usage is present, capture it
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield {"token": delta.content}

            latency_ms = (time.time() - start_time) * 1000
            yield {
                "token": "",
                "metadata": {
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms
                }
            }
        except Exception as e:
            logger.error("openai_stream_failed", error=str(e), model=model)
            raise e

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, default_model: str = "claude-3-haiku-20240307") -> None:
        key = api_key or settings.anthropic_api_key
        self.client = AsyncAnthropic(api_key=key or "dummy-key")
        self.default_model = default_model

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        model = kwargs.get("model", self.default_model)
        start_time = time.time()

        # Transform messages structure for Anthropic (system instruction is a separate param)
        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=anthropic_messages # type: ignore
            )
            latency_ms = (time.time() - start_time) * 1000

            text = response.content[0].text if response.content else ""
            prompt_tokens = response.usage.input_tokens if response.usage else 0
            completion_tokens = response.usage.output_tokens if response.usage else 0

            return LLMResponse(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms
            )
        except Exception as e:
            logger.error("anthropic_completion_failed", error=str(e), model=model)
            raise e

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        model = kwargs.get("model", self.default_model)
        start_time = time.time()

        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        try:
            prompt_tokens = 0
            completion_tokens = 0

            async with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=anthropic_messages # type: ignore
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield {"token": event.delta.text}
                    elif event.type == "message_start":
                        prompt_tokens = event.message.usage.input_tokens if event.message.usage else 0
                    elif event.type == "message_delta":
                        completion_tokens = event.usage.output_tokens if event.usage else 0

            latency_ms = (time.time() - start_time) * 1000
            yield {
                "token": "",
                "metadata": {
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms
                }
            }
        except Exception as e:
            logger.error("anthropic_stream_failed", error=str(e), model=model)
            raise e

class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: Optional[str] = None, default_model: str = "llama3.2:3b") -> None:
        self.base_url = base_url or settings.generation.ollama.base_url or "http://localhost:11434"
        self.default_model = default_model

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        model = kwargs.get("model", self.default_model)
        # Strip "ollama/" prefix if passed through API
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "")
            
        start_time = time.time()
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()

            # Parse metrics from Ollama response
            # Ollama returns eval_duration in nanoseconds
            eval_duration_ns = data.get("eval_duration", 0)
            latency_ms = eval_duration_ns / 1_000_000.0 if eval_duration_ns else (time.time() - start_time) * 1000

            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            text = data.get("message", {}).get("content", "")

            return LLMResponse(
                text=text,
                model=f"ollama/{model}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms
            )
        except Exception as e:
            logger.error("ollama_completion_failed", error=str(e), model=model, url=self.base_url)
            raise e

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        model = kwargs.get("model", self.default_model)
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "")

        start_time = time.time()
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": True
        }

        try:
            prompt_tokens = 0
            completion_tokens = 0
            latency_ms = 0.0

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        import json
                        chunk = json.loads(line)
                        
                        # Yield tokens
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield {"token": content}

                        # Check if it's the final chunk
                        if chunk.get("done", False):
                            prompt_tokens = chunk.get("prompt_eval_count", 0)
                            completion_tokens = chunk.get("eval_count", 0)
                            eval_duration_ns = chunk.get("eval_duration", 0)
                            latency_ms = eval_duration_ns / 1_000_000.0 if eval_duration_ns else (time.time() - start_time) * 1000

            yield {
                "token": "",
                "metadata": {
                    "model": f"ollama/{model}",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms
                }
            }
        except Exception as e:
            logger.error("ollama_stream_failed", error=str(e), model=model)
            raise e
