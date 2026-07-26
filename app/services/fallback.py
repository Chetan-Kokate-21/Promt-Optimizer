"""Fallback behavior for resilient prompt optimization responses."""

from __future__ import annotations

from typing import Any

from app.services.evaluator import EvaluatorService


class FallbackService:
    """Provide fallback behavior when primary optimization stages cannot run."""

    def __init__(self, evaluator_service: EvaluatorService | None = None) -> None:
        """Initialize services used to build degraded responses."""
        self.evaluator_service = evaluator_service or EvaluatorService()

    def handle(self, payload: dict[str, Any], error: Exception | None = None) -> dict[str, Any]:
        """Return a fallback result structure for degraded execution paths."""
        original_prompt = str(payload.get("prompt", "")).strip()
        cleaned_prompt = " ".join(original_prompt.split())
        mode = str(payload.get("mode", "cost")).lower() or "cost"

        return {
            "optimized_prompt": cleaned_prompt,
            "metrics": self.evaluator_service.evaluate(
                original_prompt=original_prompt,
                optimized_prompt=cleaned_prompt,
                mode=mode,
            )["metrics"],
            "pipeline_status": "fallback",
            "error": str(error) if error else "Fallback pipeline used.",
        }
