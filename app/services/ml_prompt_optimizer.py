"""Deterministic ML-guided prompt optimization based on predicted strategies.

This module turns ML strategy predictions into concrete prompt rewrites without
using randomness. It provides a middle layer between rule-based optimization
and the genetic algorithm so the system can escalate intelligently.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.domain_knowledge import get_domain_profile, infer_domain


class MLPromptOptimizerService:
    """Apply predicted prompt strategies as deterministic text transformations."""

    def optimize(
        self,
        prompt: str,
        mode: str,
        strategy_data: dict[str, Any],
        domain_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a prompt optimized using ML-recommended strategies."""
        strategies = strategy_data.get("recommended_strategies", []) or []
        working_prompt = prompt.strip() or "Provide a clear and accurate response."
        domain_context = dict(domain_context or {})
        domain = str(domain_context.get("domain") or strategy_data.get("features", {}).get("domain") or infer_domain(working_prompt)).lower()
        domain_profile = get_domain_profile(domain)
        domain_profile["expanded_terms"] = list(domain_context.get("expanded_terms") or domain_context.get("related_terms") or [])
        domain_profile["blend_phrase"] = str(domain_context.get("blend_phrase", "")).strip()

        for strategy in strategies:
            working_prompt = self._apply_strategy(working_prompt, strategy, mode, domain_profile)

        if mode == "context":
            working_prompt = self._apply_domain_context(working_prompt, domain_profile)

        working_prompt = self._normalize_prompt(working_prompt)
        return {
            "optimized_prompt": working_prompt,
            "mode": mode,
            "applied_strategies": strategies,
            "ranked_strategies": strategy_data.get("ranked_strategies", []),
            "domain_profile": {
                "domain": domain,
                "keywords": list(domain_profile.get("keywords", [])),
                "guidance": str(domain_profile.get("guidance", "")),
            },
        }

    def _apply_strategy(self, prompt: str, strategy: str, mode: str, domain_profile: dict[str, Any]) -> str:
        """Apply one predicted strategy without randomness."""
        if strategy == "direct_instruction":
            return self._ensure_section(prompt, "Task", self._extract_core_instruction(prompt))
        if strategy == "compression":
            return self._compress_prompt(prompt)
        if strategy == "remove_examples":
            return self._remove_examples(prompt)
        if strategy == "minimal_output":
            return self._ensure_section(prompt, "Output", "Return only the essential answer.")
        if strategy == "role_based":
            return self._ensure_section(prompt, "Role", str(domain_profile.get("role", self._infer_role(prompt))))
        if strategy == "few_shot":
            return self._ensure_examples(prompt, domain_profile)
        if strategy == "chain_of_thought":
            return self._ensure_section(prompt, "Reasoning", "Think through the task step by step before answering.")
        if strategy == "constraint_based":
            return self._ensure_section(prompt, "Constraints", self._default_constraint(mode, domain_profile))
        return prompt

    def _ensure_section(self, prompt: str, section: str, value: str) -> str:
        """Add a section if it is missing."""
        if re.search(rf"(?im)^{re.escape(section)}:\s*", prompt):
            return prompt
        return f"{section}: {value}\n{prompt}".strip()

    def _ensure_examples(self, prompt: str, domain_profile: dict[str, Any]) -> str:
        """Add a compact few-shot example if none exists."""
        if re.search(r"(?im)^(Examples|Example):\s*", prompt):
            return prompt
        example = f"Examples:\n- {domain_profile.get('example', 'Input: vague request | Output: precise, structured prompt')}"
        return f"{prompt}\n{example}".strip()

    def _remove_examples(self, prompt: str) -> str:
        """Remove example sections for cheaper prompts."""
        lines = prompt.splitlines()
        filtered: list[str] = []
        skipping_examples = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"^(Examples|Example):\s*", stripped):
                skipping_examples = True
                continue
            if skipping_examples and re.match(r"^(Role|Task|Goal|Context|Constraints|Output|Tone|Reasoning):\s*", stripped):
                skipping_examples = False
            if not skipping_examples:
                filtered.append(line)
        return "\n".join(filtered).strip()

    def _compress_prompt(self, prompt: str) -> str:
        """Shorten verbose content while preserving structured sections."""
        compressed_lines: list[str] = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if ":" in stripped:
                section, value = stripped.split(":", 1)
                compressed_lines.append(f"{section.strip()}: {self._shorten_value(value.strip())}")
            else:
                compressed_lines.append(self._shorten_value(stripped))
        return "\n".join(compressed_lines).strip()

    def _extract_core_instruction(self, prompt: str) -> str:
        """Extract a direct task statement from the prompt text."""
        for line in prompt.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if ":" in stripped:
                section, value = stripped.split(":", 1)
                if section.strip().lower() in {"task", "goal"} and value.strip():
                    return value.strip().rstrip(".") + "."
            return stripped.rstrip(".") + "."
        return "Provide a clear response."

    def _infer_role(self, prompt: str) -> str:
        """Infer a stable role from prompt content."""
        lowered = prompt.lower()
        if any(term in lowered for term in {"python", "code", "api", "function", "debug"}):
            return "Act as a senior software engineer."
        if any(term in lowered for term in {"math", "solve", "equation", "algebra"}):
            return "Act as a careful math tutor."
        return "Act as a helpful expert."

    def _default_constraint(self, mode: str, domain_profile: dict[str, Any]) -> str:
        """Return a mode-aware default constraint."""
        if mode == "cost":
            return "Keep the answer concise and preserve only essential detail."
        return str(domain_profile.get("constraint", "Keep the answer accurate, clear, and well-structured."))

    def _apply_domain_context(self, prompt: str, domain_profile: dict[str, Any]) -> str:
        """Inject domain-specific keywords and guidance into context-mode prompts."""
        blend_phrase = str(domain_profile.get("blend_phrase", "")).strip()
        guidance = str(domain_profile.get("guidance", "")).strip()
        if guidance:
            prompt = self._append_to_section(prompt, "Context", guidance)
        if blend_phrase:
            prompt = self._append_to_section(
                prompt,
                "Task",
                blend_phrase,
            )
        constraint = str(domain_profile.get("constraint", ""))
        if blend_phrase:
            prompt = self._append_to_section(
                prompt,
                "Constraints",
                f"Also fold in related concepts naturally {blend_phrase.replace('while covering', 'by covering')}.",
            )
            return prompt
        prompt = self._append_to_section(prompt, "Constraints", constraint)
        return prompt

    def _append_to_section(self, prompt: str, section: str, value: str) -> str:
        """Append a value to a structured section if it exists, otherwise create it."""
        pattern = rf"(?im)^({re.escape(section)}:\s*)(.*)$"
        match = re.search(pattern, prompt)
        if not value.strip():
            return prompt
        if not match:
            return self._ensure_section(prompt, section, value)
        current = match.group(2).strip()
        if value in current:
            return prompt
        updated = f"{match.group(1)}{current} {value}".strip()
        return re.sub(pattern, updated, prompt, count=1)

    def _shorten_value(self, value: str, max_words: int = 18) -> str:
        """Trim a value to a compact word budget."""
        words = value.split()
        if len(words) <= max_words:
            return value
        return " ".join(words[:max_words]).rstrip(",;:") + "."

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize spacing while preserving readability."""
        normalized = re.sub(r"[ \t]+", " ", prompt)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()
