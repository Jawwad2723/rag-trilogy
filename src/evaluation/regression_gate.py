import json
import sys
from typing import Dict, Any, Tuple
from src.logger import get_logger

logger = get_logger(__name__)

QUALITY_THRESHOLDS = {
    "faithfulness": 0.80,        # RAGAS faithfulness
    "answer_relevancy": 0.75,    # RAGAS answer relevancy
    "context_precision": 0.70,   # RAGAS context precision
    "citation_violation_rate": 0.05,  # max 5% of responses can have violations
}

def gate(results: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Checks the evaluation results against quality thresholds.
    Returns:
        (passed: bool, failures: Dict[str, str])
    """
    avg_scores = results.get("avg_scores", {})
    citation_violation_rate = results.get("citation_violation_rate", 0.0)

    failures = {}

    # Check Ragas metrics
    for metric, threshold in QUALITY_THRESHOLDS.items():
        if metric == "citation_violation_rate":
            continue
        
        score = avg_scores.get(metric, 0.0)
        if score < threshold:
            failures[metric] = f"Score {score:.4f} is below threshold {threshold:.4f}"

    # Check citation violation rate (should be <= threshold)
    max_violation = QUALITY_THRESHOLDS["citation_violation_rate"]
    if citation_violation_rate > max_violation:
        failures["citation_violation_rate"] = (
            f"Violation rate {citation_violation_rate:.4f} is above threshold {max_violation:.4f}"
        )

    passed = len(failures) == 0
    return passed, failures

def gate_from_file(filepath: str) -> None:
    """Reads eval results file and exits with code 1 if quality gate fails."""
    try:
        with open(filepath, "r") as f:
            results = json.load(f)
        
        passed, failures = gate(results)
        
        if passed:
            logger.info("quality_gate_passed", results=results.get("avg_scores"))
            print("SUCCESS: RAG Quality Gate Passed!")
            sys.exit(0)
        else:
            logger.error("quality_gate_failed", failures=failures, scores=results.get("avg_scores"))
            print("FAILURE: RAG Quality Gate Failed! Regressions detected:")
            for metric, reason in failures.items():
                print(f"  - {metric}: {reason}")
            sys.exit(1)
            
    except Exception as e:
        logger.error("quality_gate_execution_failed", error=str(e))
        print(f"ERROR: Quality gate execution failed: {str(e)}")
        sys.exit(2)
