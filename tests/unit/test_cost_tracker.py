import pytest
from src.monitoring.cost_tracker import CostTracker

def test_exact_model_pricing() -> None:
    pricing = {
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4o": {"prompt": 0.005, "completion": 0.015}
    }
    tracker = CostTracker(cost_matrix=pricing)
    
    # 2000 prompt tokens, 1000 completion tokens
    cost = tracker.calculate_cost("gpt-4o-mini", 2000, 1000)
    # 2000 * 0.00015 / 1000 = 0.0003
    # 1000 * 0.0006 / 1000 = 0.0006
    # Total = 0.0009
    assert abs(cost - 0.0009) < 1e-9

def test_wildcard_model_pricing() -> None:
    pricing = {
        "ollama/*": {"prompt": 0.0, "completion": 0.0},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006}
    }
    tracker = CostTracker(cost_matrix=pricing)
    
    cost = tracker.calculate_cost("ollama/llama3.2", 5000, 2000)
    assert cost == 0.0

def test_missing_model_fallback() -> None:
    pricing = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "*": {"prompt": 0.001, "completion": 0.002}
    }
    tracker = CostTracker(cost_matrix=pricing)
    
    # Matches fallback wildcard '*'
    cost = tracker.calculate_cost("non-existent-model", 1000, 1000)
    # 1000 * 0.001 / 1000 = 0.001
    # 1000 * 0.002 / 1000 = 0.002
    # Total = 0.003
    assert abs(cost - 0.003) < 1e-9
