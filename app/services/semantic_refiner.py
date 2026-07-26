"""Semantic refinement helpers for making prompt wording more explicit."""

from __future__ import annotations

import re
from typing import Any


class SemanticRefinerService:
    """Refine prompt semantics while preserving user intent."""

    _REPLACEMENTS = {
        r"\bpls\b": "please",
        r"\binfo\b": "information",
        r"\bctx\b": "context",
        r"\bw\/\b": "with",
        r"\be\.g\.\b": "for example",
    }

    def refine(self, prompt_data: dict[str, Any]) -> dict[str, Any]:
        """Apply lightweight semantic cleanup to prompt text."""
        refined = dict(prompt_data)
        prompt = str(prompt_data.get("prompt", ""))
        for pattern, replacement in self._REPLACEMENTS.items():
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)

        if prompt and not prompt.endswith((".", "?", "!")):
            prompt = f"{prompt}."

        refined["prompt"] = re.sub(r"\s+", " ", prompt).strip()
        refined["semantic_refinement"] = {
            "status": "applied",
            "preserved_intent": True,
        }
        return refined
