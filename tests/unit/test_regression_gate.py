"""
Unit tests for the regression gate quality gate logic.
"""
import pytest
from src.evaluation.regression_gate import gate, QUALITY_THRESHOLDS


def test_gate_passes_all_above_threshold() -> None:
    results = {
        "avg_scores": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
        },
        "citation_violation_rate": 0.02,
    }
    passed, failures = gate(results)
    assert passed is True
    assert failures == {}


def test_gate_fails_low_faithfulness() -> None:
    results = {
        "avg_scores": {
            "faithfulness": 0.70,   # Below 0.80 threshold
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
        },
        "citation_violation_rate": 0.02,
    }
    passed, failures = gate(results)
    assert passed is False
    assert "faithfulness" in failures


def test_gate_fails_high_citation_violations() -> None:
    results = {
        "avg_scores": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
        },
        "citation_violation_rate": 0.10,  # Above 5% threshold
    }
    passed, failures = gate(results)
    assert passed is False
    assert "citation_violation_rate" in failures


def test_gate_fails_multiple_metrics() -> None:
    results = {
        "avg_scores": {
            "faithfulness": 0.60,
            "answer_relevancy": 0.50,
            "context_precision": 0.40,
        },
        "citation_violation_rate": 0.20,
    }
    passed, failures = gate(results)
    assert passed is False
    assert len(failures) >= 3


def test_gate_exactly_at_threshold() -> None:
    """Scores exactly at threshold should pass (>= comparison)."""
    results = {
        "avg_scores": {
            "faithfulness": 0.80,           # Exactly at threshold
            "answer_relevancy": 0.75,       # Exactly at threshold
            "context_precision": 0.70,      # Exactly at threshold
        },
        "citation_violation_rate": 0.05,   # Exactly at max threshold
    }
    passed, failures = gate(results)
    assert passed is True


def test_gate_missing_metrics_treated_as_zero() -> None:
    """Missing metrics default to 0.0 and should fail gate."""
    results = {
        "avg_scores": {},   # No metrics
        "citation_violation_rate": 0.0,
    }
    passed, failures = gate(results)
    assert passed is False
    assert "faithfulness" in failures
    assert "answer_relevancy" in failures
