"""Decision engine for full prompt optimization comparison.

This service runs rule-based, ML-guided, and genetic optimization for each
request, evaluates every stage output, and returns the best-scoring prompt
among those candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.evaluator import EvaluatorService
from app.services.genetic_algorithm import GeneticAlgorithmService
from app.services.ml_prompt_optimizer import MLPromptOptimizerService
from app.services.rule_based_optimizer import RuleBasedPromptOptimizer
from app.services.strategy_predictor import StrategyPredictorService


@dataclass(slots=True)
class StageOutcome:
    """Represent the result of a single optimization stage."""

    stage: str
    optimized_prompt: str
    metrics: dict[str, Any]
    metadata: dict[str, Any]


class OptimizationBrainService:
    """Choose rules, judge quality, and compare outputs across all stages."""

    def __init__(
        self,
        evaluator_service: EvaluatorService,
        rule_based_optimizer: RuleBasedPromptOptimizer,
        strategy_predictor_service: StrategyPredictorService,
        ml_prompt_optimizer_service: MLPromptOptimizerService,
        genetic_algorithm_service: GeneticAlgorithmService,
    ) -> None:
        """Initialize the decision engine dependencies."""
        self.evaluator_service = evaluator_service
        self.rule_based_optimizer = rule_based_optimizer
        self.strategy_predictor_service = strategy_predictor_service
        self.ml_prompt_optimizer_service = ml_prompt_optimizer_service
        self.genetic_algorithm_service = genetic_algorithm_service

    def optimize(
        self,
        original_prompt: str,
        enriched_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run every optimization stage and choose the best final result."""
        mode = str(enriched_data.get("mode", "cost")).lower()
        selected_rules = self._select_rules(enriched_data, mode)
        rule_result = self.rule_based_optimizer.optimize(
            prompt=original_prompt,
            mode=mode,
            selected_rules=selected_rules,
            domain_context=enriched_data.get("domain_term_expansion", {}),
        )
        rule_stage = self._evaluate_stage(
            stage="rule_based",
            original_prompt=original_prompt,
            optimized_prompt=str(rule_result["optimized_prompt"]),
            mode=mode,
            metadata={
                "selected_rules": selected_rules,
                "rule_result": rule_result,
            },
        )

        ml_seed = dict(enriched_data)
        ml_seed["baseline_prompt"] = rule_stage.optimized_prompt
        strategy_data = self.strategy_predictor_service.predict(ml_seed)
        ml_result = self.ml_prompt_optimizer_service.optimize(
            prompt=rule_stage.optimized_prompt,
            mode=mode,
            strategy_data=strategy_data,
            domain_context=enriched_data.get("enrichment_metadata", {}),
        )
        ml_stage = self._evaluate_stage(
            stage="ml_guided",
            original_prompt=original_prompt,
            optimized_prompt=str(ml_result["optimized_prompt"]),
            mode=mode,
            metadata={
                "strategy_data": strategy_data,
                "ml_result": ml_result,
            },
        )

        ga_seed = dict(enriched_data)
        ga_seed["baseline_prompt"] = ml_stage.optimized_prompt
        ga_result = self.genetic_algorithm_service.optimize(
            seed_data=ga_seed,
            strategy_data=strategy_data,
        )
        ga_prompt = str(ga_result.get("best_prompt", "")).strip() or ml_stage.optimized_prompt
        ga_stage = self._evaluate_stage(
            stage="genetic",
            original_prompt=original_prompt,
            optimized_prompt=ga_prompt,
            mode=mode,
            metadata={"ga_result": ga_result},
        )
        best_final_stage = max(
            [rule_stage, ml_stage, ga_stage],
            key=lambda outcome: float(outcome.metrics.get("improvement_score", 0.0)),
        )
        return self._build_result(
            chosen_stage=best_final_stage,
            rule_stage=rule_stage,
            ml_stage=ml_stage,
            ga_stage=ga_stage,
            strategy_data=strategy_data,
        )

    def _select_rules(self, enriched_data: dict[str, Any], mode: str) -> list[str]:
        """Choose the exact rule set for the deterministic optimizer."""
        text = str(enriched_data.get("prompt", "")).lower()
        metadata = dict(enriched_data.get("metadata", {}))
        domain = str(metadata.get("domain", "general")).lower()

        if mode == "cost":
            rules = ["remove_redundant_words", "convert_to_direct_instruction"]
            if len(text.split()) > 18:
                rules.append("reduce_token_length")
            if any(term in text for term in {"complex", "detailed", "explain", "analyze"}):
                rules.insert(1, "simplify_sentences")
            return list(dict.fromkeys(rules))

        rules = ["add_role_prompting", "add_constraints_or_structure", "improve_clarity"]
        if not any(marker in text for marker in {"example", "examples", "input:", "output:"}):
            rules.append("add_examples_if_missing")
        if domain in {"coding", "math"}:
            rules.insert(1, "add_examples_if_missing")
        return list(dict.fromkeys(rules))

    def _evaluate_stage(
        self,
        stage: str,
        original_prompt: str,
        optimized_prompt: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> StageOutcome:
        """Evaluate a stage output and package its metadata."""
        evaluation = self.evaluator_service.evaluate(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            mode=mode,
        )
        return StageOutcome(
            stage=stage,
            optimized_prompt=optimized_prompt,
            metrics=evaluation["metrics"],
            metadata=metadata,
        )

    def _build_result(
        self,
        chosen_stage: StageOutcome,
        rule_stage: StageOutcome,
        ml_stage: StageOutcome,
        ga_stage: StageOutcome,
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the final decision package for the orchestrator."""
        stages: dict[str, Any] = {
            "rule_based": {
                "optimized_prompt": rule_stage.optimized_prompt,
                "metrics": rule_stage.metrics,
                **rule_stage.metadata,
            }
        }
        stages["ml_guided"] = {
            "optimized_prompt": ml_stage.optimized_prompt,
            "metrics": ml_stage.metrics,
            **ml_stage.metadata,
        }
        stages["genetic"] = {
            "optimized_prompt": ga_stage.optimized_prompt,
            "metrics": ga_stage.metrics,
            **ga_stage.metadata,
        }

        return {
            "optimized_prompt": chosen_stage.optimized_prompt,
            "metrics": chosen_stage.metrics,
            "chosen_stage": chosen_stage.stage,
            "brain_trace": {
                "stages": stages,
                "strategy_prediction": strategy_data,
            },
        }
