from config.llm.judge import LLMJudge
from config.llm.replay_exec import ReplayExecutor
from utils.util import CostCalculator, RefusalDetector, SimilarityEvaluator
from data.dataclass import EvaluationResult, ReplayResponse

from utils.portkey import get_portkey_client

class EvaluationOrchestrator:

    def __init__(
        self,
        model_configs,
        pricing_table
    ):
        # Single Portkey client - Model Catalog handles all providers
        self.client = get_portkey_client()
        self.judge = LLMJudge(self.client, model="gpt-4o-mini")
        self.cost_calc = CostCalculator()
        self.refusal_detector = RefusalDetector()
        self.similarity_evaluator = SimilarityEvaluator()
        self.model_configs = model_configs
        self.last_replay_results = {}

    def evaluate_prompt(self, prompt, reference):
        # 1️⃣ Replay across all models
        replay_results = {}
        
        for model_config in self.model_configs:
            provider = model_config["provider"]
            model_slug = model_config["slug"]
            
            try:
                # Use single client - Model Catalog routes to correct provider
                executor = ReplayExecutor(self.client)
                
                # Execute replay for this specific model
                result = executor.run_single(prompt, model_slug, provider)
                replay_results[model_slug] = result
            except Exception as e:
                print(f"Error evaluating {model_slug} from {provider}: {e}")
                # Create a fallback response
                replay_results[model_slug] = ReplayResponse(
                    model_slug=model_slug,
                    provider=provider,
                    response_text=f"Error: {str(e)}",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0
                )

        # Store for potential later use
        self.last_replay_results = replay_results

        # 2️⃣ Collect candidates for judge with input_tokens, output_tokens and other parameters.
        candidate_map = {
            m: {
                "response_text": r.response_text,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms
            }
            for m, r in replay_results.items()
        }

        # 3️⃣ Grouped judge call
        judge_output = self.judge.evaluate_group(
            prompt,
            reference,
            candidate_map
        )

        # 4️⃣ Merge per-model results
        final_results = []

        for model_slug, replay in replay_results.items():
            final_results.append(
                EvaluationResult(
                    model_slug=model_slug,
                    provider=replay.provider,
                    response_text=replay.response_text,
                    cost_usd=self.cost_calc.calculate(
                        model_slug,
                        replay.input_tokens,
                        replay.output_tokens
                    ),
                    refusal=self.refusal_detector.detect(
                        replay.response_text
                    ),
                    similarity_score=self.similarity_evaluator.score(
                        reference,
                        replay.response_text
                    ),
                    latency_ms=replay.latency_ms,
                    judge_score=judge_output["scores"].get(model_slug, 0),
                    judge_reasoning=judge_output["reasoning"]
                )
            )

        return final_results, judge_output["reasoning"]
