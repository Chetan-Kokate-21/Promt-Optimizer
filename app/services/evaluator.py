"""Evaluation utilities for prompt optimization outputs.

This module computes reusable evaluation metrics for original and optimized
prompts, including token counts, semantic similarity, output quality, and a
blended improvement score. The response shape is designed to be directly
serializable as structured JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.fitness import FitnessService


@dataclass(slots=True)
class PromptEvaluationResult:
    """Structured evaluation payload for an optimization result."""

    token_count_before: int
    token_count_after: int
    semantic_score: float
    improvement_score: float
    output_quality_before: float
    output_quality_after: float

    def to_dict(self) -> dict[str, float]:
        """Serialize evaluation metrics into a JSON-friendly dictionary."""
        return {
            "token_count_before": self.token_count_before,
            "token_count_after": self.token_count_after,
            "semantic_score": round(self.semantic_score, 4),
            "improvement_score": round(self.improvement_score, 4),
            "output_quality_before": round(self.output_quality_before, 4),
            "output_quality_after": round(self.output_quality_after, 4),
        }


class EvaluatorService:
    """Evaluate optimized prompt outputs and prepare response metadata."""

    def __init__(self, fitness_service: FitnessService | None = None) -> None:
        """Initialize reusable evaluation dependencies."""
        self.fitness_service = fitness_service or FitnessService()

    def evaluate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        mode: str = "cost",
    ) -> dict[str, Any]:
        """Return structured JSON-ready evaluation metrics for a prompt pair."""
        token_count_before = self.fitness_service.token_count(original_prompt)
        token_count_after = self.fitness_service.token_count(optimized_prompt)
        semantic_score = self.fitness_service.semantic_similarity(original_prompt, optimized_prompt)
        output_quality_before = self.fitness_service.output_quality_score(original_prompt)
        output_quality_after = self.fitness_service.output_quality_score(optimized_prompt)
        improvement_score = self.fitness_service.improvement_score(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            mode=mode,
        )

        metrics = PromptEvaluationResult(
            token_count_before=token_count_before,
            token_count_after=token_count_after,
            semantic_score=semantic_score,
            improvement_score=improvement_score,
            output_quality_before=output_quality_before,
            output_quality_after=output_quality_after,
        )

        return {
            "metrics": metrics.to_dict(),
            "summary": {
                "mode": mode,
                "token_reduction": token_count_before - token_count_after,
                "semantic_retention": round(semantic_score * 100, 2),
            },
        }
