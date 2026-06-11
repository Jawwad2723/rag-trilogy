import os
from typing import Dict, Any, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# Mapping metric name strings to Ragas metric objects
METRIC_MAPPING = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision
}

class QualityEvaluator:
    def __init__(self) -> None:
        pass

    async def evaluate_qa(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluates a single Q&A generation block using Ragas metrics.
        Returns a dict of metric scores.
        """
        # Ensure OpenAI API key is set for Ragas (as Ragas defaults to OpenAI for evaluation)
        if not os.getenv("OPENAI_API_KEY") and settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        try:
            dataset = Dataset.from_dict(data)
            
            # Resolve metrics to use from configuration
            metrics_to_use = []
            for name in settings.evaluation.ragas_metrics:
                metric_obj = METRIC_MAPPING.get(name)
                # For some metrics (like context_precision), ground_truth might be required
                if name == "context_precision" and not ground_truth:
                    logger.warning("skipping_context_precision_no_ground_truth")
                    continue
                if metric_obj:
                    metrics_to_use.append(metric_obj)

            if not metrics_to_use:
                return {}

            logger.info("running_ragas_evaluation", metrics=[m.name for m in metrics_to_use])
            result = evaluate(
                dataset,
                metrics=metrics_to_use
            )
            
            scores = {k: float(v) for k, v in result.items() if not isinstance(v, list)}
            logger.info("ragas_evaluation_completed", scores=scores)
            return scores
            
        except Exception as e:
            logger.error("ragas_evaluation_failed", error=str(e))
            # Graceful fallback: return zeroed metrics or empty dict
            return {name: 0.0 for name in settings.evaluation.ragas_metrics}
from typing import Optional
