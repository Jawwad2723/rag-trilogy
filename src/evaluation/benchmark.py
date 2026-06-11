import asyncio
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from src.generation.chain import RAGChain
from src.evaluation.dataset import load_golden_dataset
from src.monitoring.quality_metrics import QualityEvaluator
from src.logger import get_logger

logger = get_logger(__name__)

class ModelBenchmarker:
    def __init__(self, chain: RAGChain, models: List[str] | None = None) -> None:
        self.chain = chain
        self.models = models or ["llama3.2:3b", "mistral:7b", "phi3:mini"]
        self.quality_evaluator = QualityEvaluator()

    async def run_benchmark(
        self,
        dataset_path: str = "tests/eval/golden_dataset.json",
        num_questions: int = 20
    ) -> Dict[str, Any]:
        """
        Runs the benchmark suite comparing different Ollama models.
        """
        dataset = load_golden_dataset(dataset_path)
        if not dataset:
            logger.warning("empty_dataset_for_benchmark")
            return {}
        
        # Take a slice of questions
        test_questions = dataset[:num_questions]
        
        results: Dict[str, List[Dict[str, Any]]] = {model: [] for model in self.models}
        
        logger.info("starting_models_benchmark", models=self.models, questions=len(test_questions))

        for model in self.models:
            ollama_model_id = f"ollama/{model}"
            logger.info("benchmarking_model", model=ollama_model_id)

            for idx, item in enumerate(test_questions):
                question = item["question"]
                ground_truth = item.get("ground_truth", "")

                start_time = time.time()
                ttft = 0.0
                tokens_count = 0
                full_text = []

                # Let's run a streaming query to compute TTFT
                try:
                    stream_generator = self.chain.stream_query(
                        question=question,
                        model=ollama_model_id,
                        top_k=5
                    )
                    
                    async for chunk in stream_generator:
                        if "token" in chunk and chunk["token"]:
                            if ttft == 0.0:
                                ttft = (time.time() - start_time) * 1000
                            full_text.append(chunk["token"])
                            tokens_count += 1
                        
                        if "metadata" in chunk:
                            meta = chunk["metadata"]
                            completion_tokens = meta.get("completion_tokens", 0)
                            prompt_tokens = meta.get("prompt_tokens", 0)
                            gen_latency = meta.get("generation_latency_ms", 1.0)
                            total_latency = meta.get("total_latency_ms", 1.0)
                            citations = meta.get("citations", [])

                    answer = "".join(full_text)
                    tokens_per_sec = completion_tokens / (gen_latency / 1000.0) if gen_latency > 0 else 0.0

                    # Evaluate Faithfulness & Recall
                    contexts = [c["text_preview"] for c in citations]
                    eval_scores = await self.quality_evaluator.evaluate_qa(
                        question=question,
                        answer=answer,
                        contexts=contexts,
                        ground_truth=ground_truth
                    )

                    results[model].append({
                        "ttft_ms": ttft,
                        "tokens_per_sec": tokens_per_sec,
                        "total_latency_ms": total_latency,
                        "faithfulness": eval_scores.get("faithfulness", 0.0),
                        "context_precision": eval_scores.get("context_precision", 0.0)
                    })
                    logger.info("benchmark_query_completed", model=model, q_idx=idx, latency_ms=total_latency)

                except Exception as e:
                    logger.error("benchmark_query_failed", model=model, question=question, error=str(e))
                    # Add dummy/failed records so sizes match
                    results[model].append({
                        "ttft_ms": 0.0,
                        "tokens_per_sec": 0.0,
                        "total_latency_ms": 0.0,
                        "faithfulness": 0.0,
                        "context_precision": 0.0
                    })

        # Process statistics per model
        summary = []
        for model in self.models:
            runs = results[model]
            valid_runs = [r for r in runs if r["total_latency_ms"] > 0]
            
            if not valid_runs:
                summary.append({
                    "model": model,
                    "avg_tps": 0.0,
                    "p50_lat": "0.0s",
                    "p95_lat": "0.0s",
                    "faithfulness": 0.0,
                    "context_recall": 0.0,
                    "verdict": "Failed"
                })
                continue

            # Calculate metrics
            avg_tps = sum(r["tokens_per_sec"] for r in valid_runs) / len(valid_runs)
            latencies = sorted(r["total_latency_ms"] for r in valid_runs)
            p50_lat = latencies[len(latencies) // 2] / 1000.0
            p95_lat = latencies[int(len(latencies) * 0.95)] / 1000.0
            avg_faithfulness = sum(r["faithfulness"] for r in valid_runs) / len(valid_runs)
            avg_precision = sum(r["context_precision"] for r in valid_runs) / len(valid_runs)

            verdict = "OK"
            if avg_faithfulness > 0.8:
                verdict = "Balanced" if model == "mistral:7b" else "Good trade"
            elif avg_tps > 40:
                verdict = "Fast/OK"

            summary.append({
                "model": model,
                "avg_tps": round(avg_tps, 1),
                "p50_lat": f"{p50_lat:.1f}s",
                "p95_lat": f"{p95_lat:.1f}s",
                "faithfulness": round(avg_faithfulness, 2),
                "context_recall": round(avg_precision, 2), # Using precision as proxy
                "verdict": verdict
            })

        await self._write_markdown_report(summary)
        return {"summary": summary, "details": results}

    async def _write_markdown_report(self, summary: List[Dict[str, Any]]) -> None:
        """Writes the benchmark results markdown report to the benchmarks/ directory."""
        os.makedirs("benchmarks", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"benchmarks/results_{timestamp}.md"

        lines = [
            "# Model Benchmark Results",
            f"Run Timestamp: {datetime.now().isoformat()}",
            "",
            "| Model        | Avg t/s | p50 lat | p95 lat | Faithfulness | Context Recall | Verdict     |",
            "|--------------|---------|---------|---------|--------------|----------------|-------------|"
        ]

        for s in summary:
            lines.append(
                f"| {s['model']:12} | {s['avg_tps']:7} | {s['p50_lat']:7} | {s['p95_lat']:7} | {s['faithfulness']:12} | {s['context_recall']:14} | {s['verdict']:11} |"
            )

        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        logger.info("benchmark_report_written", path=report_path)
