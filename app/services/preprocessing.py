"""Preprocessing utilities for normalizing raw optimization requests."""

from __future__ import annotations

import re
from typing import Any


class PreprocessingService:
    """Prepare incoming prompt payloads for downstream optimization steps."""

    _SUPPORTED_MODES = {"cost", "context"}

    def preprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize prompt text, mode, and optional nested request fields."""
        prompt = self._normalize_text(str(payload.get("prompt", "")))
        mode = str(payload.get("mode", "cost")).strip().lower()
        context = self._ensure_dict(payload.get("context", {}))
        constraints = self._ensure_dict(payload.get("constraints", {}))
        metadata = self._ensure_dict(payload.get("metadata", {}))

        return {
            "prompt": prompt,
            "mode": mode if mode in self._SUPPORTED_MODES else "cost",
            "context": context,
            "constraints": constraints,
            "metadata": metadata,
        }

    def _normalize_text(self, text: str) -> str:
        """Collapse whitespace while preserving human-readable spacing."""
        collapsed = re.sub(r"\s+", " ", text or "").strip()
        return collapsed

    def _ensure_dict(self, value: Any) -> dict[str, Any]:
        """Return a dictionary for nested fields, or an empty mapping."""
        return value if isinstance(value, dict) else {}
