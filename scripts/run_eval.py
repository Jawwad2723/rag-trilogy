import argparse
import asyncio
import json
import os
import sys

# Ensure src/ package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.deps import get_rag_chain
from src.evaluation.ragas_eval import RagasEvaluator
from src.logger import get_logger

logger = get_logger(__name__)

async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the golden dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Path to save evaluation results JSON."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions evaluated."
    )
    args = parser.parse_args()

    print("Initializing RAG chain for evaluation...")
    chain = get_rag_chain()
    evaluator = RagasEvaluator()

    print("Running evaluations. This may take some time depending on dataset size...")
    try:
        results = await evaluator.run_evaluation(chain, limit=args.limit)
        
        # Save results to disk
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nEvaluation completed. Results saved to {args.output}")
        print("Average Scores:")
        for metric, score in results["avg_scores"].items():
            print(f"  - {metric}: {score:.4f}")
        print(f"Citation Violation Rate: {results['citation_violation_rate']:.4f}")
        print(f"Total Duration: {results['duration_seconds']} seconds")

    except Exception as e:
        print(f"\nError running evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
