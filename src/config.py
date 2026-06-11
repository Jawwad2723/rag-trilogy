import os
import re
from typing import Dict, List, Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class AppConfig(BaseModel):
    name: str = "ask-my-docs"
    version: str = "1.0.0"
    log_level: str = "INFO"

class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    collection_name: str = "documents"

class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536

class HybridConfig(BaseModel):
    alpha: float = 0.5
    rrf_k: int = 60
    top_k: int = 20

class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5

class RetrievalConfig(BaseModel):
    vector_store: VectorStoreConfig
    embedding: EmbeddingConfig
    hybrid: HybridConfig
    reranker: RerankerConfig

class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2:3b"

class GenerationConfig(BaseModel):
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.1
    ollama: OllamaConfig

class TokenCost(BaseModel):
    prompt: float
    completion: float

class MonitoringConfig(BaseModel):
    otel_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 9090
    cost_per_1k_tokens: Dict[str, TokenCost] = {}

class EvaluationConfig(BaseModel):
    ragas_metrics: List[str] = ["faithfulness", "answer_relevancy", "context_precision"]
    thresholds: Dict[str, float] = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_precision": 0.70
    }

class Settings(BaseSettings):
    app: AppConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    monitoring: MonitoringConfig
    evaluation: EvaluationConfig

    # API keys and direct environment overrides
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

def load_settings(config_path: str = "config.yaml") -> Settings:
    # Read the YAML file if it exists, replace env variable placeholders, and load
    if not os.path.exists(config_path):
        # Search parent directories as fallback (for tests/scripts run from different dirs)
        for i in range(3):
            candidate = os.path.join("../" * (i + 1), config_path)
            if os.path.exists(candidate):
                config_path = candidate
                break

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            content = f.read()
            
        # Replace ${VAR} or ${VAR:default}
        # First check for matches
        pattern = re.compile(r'\$\{(\w+)\}')
        
        # We also need defaults. Let's map env variables to actual values or fallback to default
        fallbacks = {
            "QDRANT_URL": "http://localhost:6333",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"
        }
        
        def repl(match: re.Match[str]) -> str:
            var_name = match.group(1)
            val = os.getenv(var_name)
            if val is not None:
                return val
            return fallbacks.get(var_name, "")
            
        resolved_content = pattern.sub(repl, content)
        yaml_data = yaml.safe_load(resolved_content)
    else:
        yaml_data = {}

    # Merge yaml_data into the environment-based Settings instantiation
    return Settings(**yaml_data)

# Singleton configuration instance
settings = load_settings()
