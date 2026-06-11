import json
import os
from typing import List, Dict, Any
from src.logger import get_logger

logger = get_logger(__name__)

def load_golden_dataset(path: str = "tests/eval/golden_dataset.json") -> List[Dict[str, Any]]:
    """Loads the golden evaluation dataset from JSON file."""
    # Handle path search relative to execution directory
    if not os.path.exists(path):
        for i in range(3):
            candidate = os.path.join("../" * (i + 1), path)
            if os.path.exists(candidate):
                path = candidate
                break

    if not os.path.exists(path):
        logger.warning("golden_dataset_not_found_returning_empty", path=path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info("golden_dataset_loaded", path=path, count=len(dataset))
        return dataset
    except Exception as e:
        logger.error("golden_dataset_load_failed", path=path, error=str(e))
        raise e

def save_golden_dataset(dataset: List[Dict[str, Any]], path: str = "tests/eval/golden_dataset.json") -> None:
    """Saves the golden evaluation dataset to JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        logger.info("golden_dataset_saved", path=path, count=len(dataset))
    except Exception as e:
        logger.error("golden_dataset_save_failed", path=path, error=str(e))
        raise e
