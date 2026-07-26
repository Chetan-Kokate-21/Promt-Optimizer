"""High-level orchestration skeleton for the prompt optimization workflow."""

from __future__ import annotations

import re
from typing import Any

from app.services.context_analyzer import ContextAnalyzerService
from app.services.domain_term_expander import DomainTermExpansionService
from app.services.enrichment import EnrichmentService
from app.services.evaluator import EvaluatorService
from app.services.fallback import FallbackService
from app.services.fitness import FitnessService
from app.services.genetic_algorithm import GeneticAlgorithmService
from app.services.ml_prompt_optimizer import MLPromptOptimizerService
from app.services.optimization_brain import OptimizationBrainService
from app.services.preprocessing import PreprocessingService
from app.services.prompt_formatter import PromptFormatterService
from app.services.rule_based_optimizer import RuleBasedPromptOptimizer
from app.services.semantic_refiner import SemanticRefinerService
from app.services.strategy_predictor import StrategyPredictorService


class PromptOptimizerService:
    """Coordinate the end-to-end prompt optimization pipeline."""

    def __init__(self) -> None:
        """Initialize service dependencies for pipeline orchestration."""
        self.preprocessing_service = PreprocessingService()
        self.semantic_refiner_service = SemanticRefinerService()
        self.context_analyzer_service = ContextAnalyzerService()
        self.domain_term_expansion_service = DomainTermExpansionService()
        self.rule_based_optimizer = RuleBasedPromptOptimizer()
        self.enrichment_service = EnrichmentService()
        self.prompt_formatter_service = PromptFormatterService()
        self.strategy_predictor_service = StrategyPredictorService()
        self.ml_prompt_optimizer_service = MLPromptOptimizerService()
        self.fitness_service = FitnessService()
        self.evaluator_service = EvaluatorService()
        self.genetic_algorithm_service = GeneticAlgorithmService(
            fitness_service=self.fitness_service
        )
        self.optimization_brain_service = OptimizationBrainService(
            evaluator_service=self.evaluator_service,
            rule_based_optimizer=self.rule_based_optimizer,
            strategy_predictor_service=self.strategy_predictor_service,
            ml_prompt_optimizer_service=self.ml_prompt_optimizer_service,
            genetic_algorithm_service=self.genetic_algorithm_service,
        )
        self.fallback_service = FallbackService(
            evaluator_service=self.evaluator_service
        )

    def optimize_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the full optimization pipeline and return prompt + metrics."""
        try:
            preprocessed = self.preprocessing_service.preprocess(payload)
            self._validate_request(preprocessed)
            refined = self.semantic_refiner_service.refine(preprocessed)
            analyzed = self.context_analyzer_service.analyze(refined)
            expanded = self.domain_term_expansion_service.expand(analyzed)
            enriched = self.enrichment_service.enrich(expanded)
            decision = self.optimization_brain_service.optimize(
                original_prompt=str(preprocessed.get("prompt", "")),
                enriched_data=enriched,
            )
            optimized_prompt = self.prompt_formatter_service.format_prompt(
                prompt=decision.get("optimized_prompt", ""),
                mode=str(preprocessed.get("mode", "cost")).lower(),
                domain=str(expanded.get("domain_term_expansion", {}).get("domain", "general")),
                domain_term_expansion=expanded.get("domain_term_expansion", {}),
            )
            optimized_prompt = self._clean_prompt(optimized_prompt)
            evaluation_metrics = decision.get("metrics", {})
        except Exception as error:
            return self.fallback_service.handle(payload=payload, error=error)

        return {
            "optimized_prompt": optimized_prompt,
            "metrics": evaluation_metrics,
            "pipeline_status": "success",
            "details": {
                "preprocessed_input": preprocessed,
                "context_analysis": analyzed.get("context_analysis", {}),
                "domain_term_expansion": expanded.get("domain_term_expansion", {}),
                "brain_decision": decision,
            },
        }

    def _validate_request(self, payload: dict[str, Any]) -> None:
        """Validate required request fields before running the pipeline."""
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("The 'prompt' field is required.")

        mode = str(payload.get("mode", "cost")).lower()
        if mode not in {"cost", "context"}:
            raise ValueError("The 'mode' field must be either 'cost' or 'context'.")

    def _clean_prompt(self, prompt: str) -> str:
        """Normalize the final optimized prompt before returning it."""
        compact = re.sub(r"[ \t]+", " ", prompt or "")
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        return compact.strip()
