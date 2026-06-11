from typing import Dict, Any
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

class CostTracker:
    def __init__(self, cost_matrix: Dict[str, Any] | None = None) -> None:
        # Load from settings if not passed
        self.cost_matrix = cost_matrix or {}
        if not self.cost_matrix and settings.monitoring and settings.monitoring.cost_per_1k_tokens:
            # Pydantic parses this into Dict[str, TokenCost]
            self.cost_matrix = {
                k: {"prompt": v.prompt, "completion": v.completion}
                for k, v in settings.monitoring.cost_per_1k_tokens.items()
            }

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculates the cost of a request based on prompt & completion token counts and the model used.
        Returns the cost in USD.
        """
        prompt_rate = 0.0
        completion_rate = 0.0

        # Try to find an exact match first
        model_pricing = self.cost_matrix.get(model)

        if not model_pricing:
            # Check for prefixes like "ollama/" or wildcard matching "ollama/*"
            if model.startswith("ollama/") or "/" not in model and any(model.startswith(p.split("/")[0]) for p in self.cost_matrix if "/*" in p):
                # Look for matching pattern
                for pattern, price in self.cost_matrix.items():
                    if pattern.endswith("/*"):
                        prefix = pattern[:-2]
                        if model.startswith(prefix) or model == prefix:
                            model_pricing = price
                            break
            else:
                # If still not found, check if a general fallback pattern exists (e.g. wildcard "*")
                model_pricing = self.cost_matrix.get("*")

        if model_pricing:
            prompt_rate = model_pricing.get("prompt", 0.0)
            completion_rate = model_pricing.get("completion", 0.0)
        else:
            # Log warning if price mapping is missing
            logger.warning("missing_model_cost_pricing", model=model)

        # Cost calculation: rate is per 1,000 tokens
        cost = ((prompt_tokens * prompt_rate) + (completion_tokens * completion_rate)) / 1000.0
        return float(cost)
