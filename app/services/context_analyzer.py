"""Context analysis helpers for deriving mode and domain hints."""

from __future__ import annotations

from typing import Any

from app.services.domain_knowledge import infer_domain

class ContextAnalyzerService:
    """Inspect contextual signals that influence prompt optimization."""

    def analyze(self, prompt_data: dict[str, Any]) -> dict[str, Any]:
        """Extract contextual features from the prompt payload."""
        analyzed = dict(prompt_data)
        prompt = str(prompt_data.get("prompt", "")).lower()
        metadata = dict(prompt_data.get("metadata", {}))
        context = dict(prompt_data.get("context", {}))

        analyzed_context = {
            "domain": metadata.get("domain") or self._infer_domain(prompt),
            "intent": metadata.get("intent") or self._infer_intent(prompt),
            "has_examples": bool(context.get("examples")),
            "mode": prompt_data.get("mode", "cost"),
        }

        metadata.update(
            {
                "domain": analyzed_context["domain"],
                "intent": analyzed_context["intent"],
            }
        )
        analyzed["metadata"] = metadata
        analyzed["context_analysis"] = analyzed_context
        return analyzed

    def _infer_domain(self, prompt: str) -> str:
        """Infer a broad problem domain from plain text."""
        return infer_domain(prompt)

    def _infer_intent(self, prompt: str) -> str:
        """Infer user intent from simple keyword matching."""
        if any(term in prompt for term in {"explain", "describe", "teach"}):
            return "explain"
        if any(term in prompt for term in {"solve", "fix", "calculate"}):
            return "solve"
        if any(term in prompt for term in {"summarize", "compress"}):
            return "summarize"
        if any(term in prompt for term in {"optimize", "improve", "refine"}):
            return "optimize"
        return "generate"
