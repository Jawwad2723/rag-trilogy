import argparse
import asyncio
import os
import sys

# Ensure src/ package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.deps import get_rag_chain
from src.evaluation.benchmark import ModelBenchmarker
from src.logger import get_logger

logger = get_logger(__name__)

async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Ollama models on latency and quality.")
    parser.add_argument(
        "--questions",
        type=int,
        default=20,
        help="Number of questions from golden dataset to run in the benchmark."
    )
    args = parser.parse_args()

    print("Initializing benchmark suite...")
    chain = get_rag_chain()
    
    # We benchmark: llama3.2:3b, mistral:7b, phi3:mini
    benchmarker = ModelBenchmarker(chain=chain)

    print(f"Running model comparison on {args.questions} questions...")
    try:
        results = await benchmarker.run_benchmark(num_questions=args.questions)
        
        print("\nBenchmark successfully completed!")
        print("Results Table:")
        print("| Model        | Avg t/s | p50 lat | p95 lat | Faithfulness | Context Recall | Verdict     |")
        print("|--------------|---------|---------|---------|--------------|----------------|-------------|")
        for s in results["summary"]:
            print(
                f"| {s['model']:12} | {s['avg_tps']:7} | {s['p50_lat']:7} | {s['p95_lat']:7} | {s['faithfulness']:12} | {s['context_recall']:14} | {s['verdict']:11} |"
            )
        print("\nDetailed markdown report written to the benchmarks/ directory.")

    except Exception as e:
        print(f"\nError running model benchmark: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
