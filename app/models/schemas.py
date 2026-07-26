"""Schema skeletons for prompt optimization requests and responses."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OptimizePromptRequest:
    """Represent the incoming payload for prompt optimization."""

    prompt: str = ""
    mode: str = "cost"
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OptimizePromptRequest":
        """Build a request schema from a dictionary payload."""
        return cls(
            prompt=payload.get("prompt", ""),
            mode=payload.get("mode", "cost"),
            context=payload.get("context", {}),
            constraints=payload.get("constraints", {}),
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class OptimizePromptResponse:
    """Represent the eventual response shape for optimized prompt output."""

    optimized_prompt: str = ""
    score: float | None = None
    strategy: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
