"""Deterministic rule-based prompt optimization for baseline generation.

This module produces a guaranteed human-readable baseline optimization for a
prompt before any stochastic search is applied. It supports two modes:

- cost: compresses and clarifies prompts into direct, efficient instructions
- context: enriches prompts with role, structure, examples, and clarity cues

The implementation is deterministic and designed to plug directly into the
genetic algorithm as a stable seed candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.services.domain_knowledge import get_domain_profile, infer_domain


@dataclass(slots=True)
class RuleBasedOptimizationResult:
    """Structured result for a deterministic rule-based optimization pass."""

    optimized_prompt: str
    mode: str
    applied_rules: list[str]

    def to_dict(self) -> dict[str, object]:
        """Serialize the optimization result into a JSON-friendly dictionary."""
        return {
            "optimized_prompt": self.optimized_prompt,
            "mode": self.mode,
            "applied_rules": self.applied_rules,
        }


class RuleBasedPromptOptimizer:
    """Generate deterministic baseline prompt optimizations."""

    _REDUNDANT_PHRASES = (
        ("please", ""),
        ("kindly", ""),
        ("i want you to", ""),
        ("can you", ""),
        ("could you", ""),
        ("would you", ""),
        ("in order to", "to"),
        ("make sure to", "ensure"),
        ("it is important to note that", ""),
        ("basically", ""),
        ("actually", ""),
        ("just", ""),
        ("very", ""),
        ("really", ""),
        ("somewhat", ""),
    )

    _SIMPLIFY_MAP = (
        ("utilize", "use"),
        ("assistance", "help"),
        ("approximately", "about"),
        ("demonstrate", "show"),
        ("additional", "more"),
        ("numerous", "many"),
        ("therefore", "so"),
        ("however", "but"),
        ("regarding", "about"),
        ("provide", "give"),
    )

    def optimize(
        self,
        prompt: str,
        mode: str,
        selected_rules: list[str] | None = None,
        domain_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Return a deterministic baseline optimized prompt."""
        normalized_mode = mode.lower().strip()
        if normalized_mode not in {"cost", "context"}:
            normalized_mode = "cost"

        cleaned_prompt = self._normalize_whitespace(prompt)
        if not cleaned_prompt:
            cleaned_prompt = "Provide a clear and accurate response."

        if normalized_mode == "cost":
            result = self._optimize_for_cost(cleaned_prompt, selected_rules=selected_rules)
        else:
            result = self._optimize_for_context(
                cleaned_prompt,
                selected_rules=selected_rules,
                domain_context=domain_context,
            )
        return result.to_dict()

    def _optimize_for_cost(
        self,
        prompt: str,
        selected_rules: list[str] | None = None,
    ) -> RuleBasedOptimizationResult:
        """Apply deterministic compression and direct-instruction rules."""
        default_rules = [
            "remove_redundant_words",
            "simplify_sentences",
            "convert_to_direct_instruction",
            "reduce_token_length",
        ]
        applied_rules = selected_rules or default_rules
        compact = prompt
        for rule in applied_rules:
            compact = self._apply_rule(compact, rule, mode="cost")
        compact = self._clean_instruction(compact)
        return RuleBasedOptimizationResult(
            optimized_prompt=compact or "Respond directly and clearly.",
            mode="cost",
            applied_rules=applied_rules,
        )

    def _optimize_for_context(
        self,
        prompt: str,
        selected_rules: list[str] | None = None,
        domain_context: dict[str, object] | None = None,
    ) -> RuleBasedOptimizationResult:
        """Apply deterministic enrichment rules for more guided prompting."""
        default_rules = [
            "add_role_prompting",
            "add_examples_if_missing",
            "add_constraints_or_structure",
            "improve_clarity",
        ]
        applied_rules = selected_rules or default_rules
        domain_context = dict(domain_context or {})
        domain = str(domain_context.get("domain") or infer_domain(prompt)).lower()
        domain_profile = get_domain_profile(domain)
        blend_phrase = str(domain_context.get("blend_phrase", "")).strip()
        optimized_prompt = self._normalize_terminal_punctuation(prompt)
        for rule in applied_rules:
            optimized_prompt = self._apply_rule(
                optimized_prompt,
                rule,
                mode="context",
                domain_profile=domain_profile,
            )
        optimized_prompt = self._blend_task_context(optimized_prompt, blend_phrase)
        optimized_prompt = self._append_to_section(
            optimized_prompt,
            "Context",
            self._build_domain_context(domain_profile),
        )
        optimized_prompt = self._append_to_section(
            optimized_prompt,
            "Constraints",
            self._build_blend_guidance(blend_phrase),
        )
        if not self._has_example(optimized_prompt):
            optimized_prompt = self._append_section(
                optimized_prompt,
                "Example",
                str(domain_profile.get("example", "Input: unclear request | Output: precise, context-rich response.")),
            )
        optimized_prompt = self._finalize_context_prompt(optimized_prompt)
        return RuleBasedOptimizationResult(
            optimized_prompt=optimized_prompt,
            mode="context",
            applied_rules=applied_rules,
        )

    def _apply_rule(
        self,
        text: str,
        rule: str,
        mode: str,
        domain_profile: dict[str, object] | None = None,
    ) -> str:
        """Apply one named rule deterministically."""
        domain_profile = domain_profile or get_domain_profile("general")
        if rule == "remove_redundant_words":
            return self._remove_redundant_words(text)
        if rule == "simplify_sentences":
            return self._simplify_wording(text)
        if rule == "convert_to_direct_instruction":
            return self._convert_to_direct_instruction(text)
        if rule == "reduce_token_length":
            return self._shorten_sentences(text)
        if rule == "add_role_prompting":
            return self._ensure_context_section(text, "Role", str(domain_profile.get("role", self._infer_role(text))))
        if rule == "add_examples_if_missing":
            if self._has_example(text):
                return text
            return self._append_section(
                text,
                "Example",
                str(domain_profile.get("example", "Input: unclear request | Output: precise, context-rich response.")),
            )
        if rule == "add_constraints_or_structure":
            base_instruction = self._extract_unstructured_instruction(text)
            structured = self._ensure_context_section(
                text,
                "Task",
                self._convert_to_direct_instruction(base_instruction),
            )
            structured = self._ensure_context_section(
                structured,
                "Constraints",
                f"Keep the answer accurate, well-structured, and easy to follow. {domain_profile.get('constraint', '')}".strip(),
            )
            return self._ensure_context_section(
                structured,
                "Output",
                "Use clear sections or bullet points when helpful.",
            )
        if rule == "improve_clarity":
            clarified = self._simplify_wording(text)
            return self._normalize_terminal_punctuation(clarified) if mode == "cost" else clarified
        return text

    def _remove_redundant_words(self, text: str) -> str:
        """Remove filler phrases while preserving instruction meaning."""
        optimized = f" {text.lower()} "
        for source, target in self._REDUNDANT_PHRASES:
            optimized = optimized.replace(f" {source} ", f" {target} ")
        return self._restore_sentence_case(self._normalize_whitespace(optimized))

    def _simplify_wording(self, text: str) -> str:
        """Replace verbose wording with simpler deterministic alternatives."""
        simplified = text
        for source, target in self._SIMPLIFY_MAP:
            simplified = re.sub(rf"\b{re.escape(source)}\b", target, simplified, flags=re.IGNORECASE)
        return self._normalize_whitespace(simplified)

    def _convert_to_direct_instruction(self, text: str) -> str:
        """Convert question-style prompts into direct actionable instructions."""
        instruction = text.strip()
        instruction = re.sub(r"^[Ww](ould|ill|hat|hy|hen|here)\b.*?:\s*", "", instruction)
        instruction = re.sub(r"^[Cc]an you\s+", "", instruction)
        instruction = re.sub(r"^[Cc]ould you\s+", "", instruction)
        instruction = re.sub(r"^[Pp]lease\s+", "", instruction)
        instruction = instruction.rstrip("?.!")
        if not instruction:
            return "Provide a clear response"
        if not re.match(r"^(Explain|Write|Create|Summarize|Optimize|Solve|Analyze|Generate|Describe|Provide|List)\b", instruction, flags=re.IGNORECASE):
            instruction = f"Provide {instruction[0].lower() + instruction[1:]}" if len(instruction) > 1 else "Provide a clear response"
        return self._normalize_terminal_punctuation(instruction)

    def _shorten_sentences(self, text: str) -> str:
        """Keep the instruction concise by removing low-value clauses."""
        clauses = re.split(r",|;|\band\b|\bbut\b", text)
        kept = [clause.strip() for clause in clauses if clause.strip()]
        if not kept:
            return text
        primary = kept[0]
        if len(primary.split()) < 6 and len(kept) > 1:
            primary = f"{primary} {kept[1]}"
        return self._normalize_terminal_punctuation(primary)

    def _clean_instruction(self, text: str) -> str:
        """Normalize punctuation and ensure a valid final instruction."""
        cleaned = self._normalize_whitespace(text)
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
        cleaned = cleaned.strip(" ,;:")
        return self._normalize_terminal_punctuation(cleaned)

    def _infer_role(self, prompt: str) -> str:
        """Infer a stable role label from prompt content."""
        lowered = prompt.lower()
        if any(term in lowered for term in {"code", "python", "api", "debug", "function"}):
            return "Act as a senior software engineer."
        if any(term in lowered for term in {"math", "equation", "solve", "algebra", "calculus"}):
            return "Act as a careful math tutor."
        if any(term in lowered for term in {"write", "story", "article", "content"}):
            return "Act as a clear and concise writing assistant."
        return "Act as a helpful subject-matter expert."

    def _ensure_context_section(self, text: str, section: str, value: str) -> str:
        """Insert a structured section when it is missing."""
        if re.search(rf"(?im)^{re.escape(section)}:\s*", text):
            return text
        prefix = f"{section}: {value}".strip()
        return f"{prefix}\n{text}".strip()

    def _append_section(self, text: str, section: str, value: str) -> str:
        """Append a structured section to the end of the prompt."""
        if re.search(rf"(?im)^{re.escape(section)}:\s*", text):
            return text
        return f"{text}\n{section}: {value}".strip()

    def _append_to_section(self, text: str, section: str, value: str) -> str:
        """Append extra guidance to an existing section or create it if needed."""
        if not value.strip():
            return text
        pattern = rf"(?im)^({re.escape(section)}:\s*)(.*)$"
        match = re.search(pattern, text)
        if not match:
            return self._ensure_context_section(text, section, value)
        current = match.group(2).strip()
        if value in current:
            return text
        updated = f"{match.group(1)}{current} {value}".strip()
        return re.sub(pattern, updated, text, count=1)

    def _has_example(self, prompt: str) -> bool:
        """Detect whether the prompt already contains example-like structure."""
        lowered = prompt.lower()
        return any(marker in lowered for marker in {"example:", "examples:", "for example", "input:", "output:"})

    def _normalize_terminal_punctuation(self, text: str) -> str:
        """Ensure the prompt ends with a period for readability."""
        cleaned = self._normalize_whitespace(text).rstrip("?.!")
        return f"{cleaned}." if cleaned else "Provide a clear response."

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace while preserving line breaks."""
        normalized = re.sub(r"[ \t]+", " ", text or "")
        normalized = re.sub(r"\n[ \t]+", "\n", normalized)
        normalized = re.sub(r"\n\s*\n+", "\n", normalized)
        return normalized.strip()

    def _restore_sentence_case(self, text: str) -> str:
        """Restore sentence-style capitalization after lowercase processing."""
        if not text:
            return text
        return text[0].upper() + text[1:]

    def _extract_unstructured_instruction(self, text: str) -> str:
        """Extract the non-section prompt content from structured text."""
        content_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^(Role|Task|Constraints|Output|Example|Examples):\s*", stripped):
                continue
            content_lines.append(stripped)
        if content_lines:
            return " ".join(content_lines)
        return text

    def _finalize_context_prompt(self, text: str) -> str:
        """Remove duplicated freeform lines once structured sections exist."""
        task_value = ""
        cleaned_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("task:"):
                task_value = stripped.split(":", 1)[1].strip().rstrip(".")
                cleaned_lines.append(stripped)
                continue
            if ":" not in stripped and task_value and stripped.rstrip(".") == task_value:
                continue
            cleaned_lines.append(stripped)
        return self._normalize_whitespace("\n".join(cleaned_lines))

    def _build_domain_context(self, domain_profile: dict[str, object]) -> str:
        """Build a compact domain-aware context line."""
        return str(domain_profile.get("guidance", "")).strip()

    def _build_blend_guidance(self, blend_phrase: str) -> str:
        """Build only the extra related-concepts guidance to avoid duplication."""
        if not blend_phrase:
            return ""
        return f"Also enrich the response naturally {blend_phrase.replace('while covering', 'by covering')}."

    def _blend_task_context(self, prompt: str, blend_phrase: str) -> str:
        """Blend related concepts into the task section rather than adding a separate list."""
        if not blend_phrase:
            return prompt
        pattern = r"(?im)^(Task:\s*)(.*)$"
        match = re.search(pattern, prompt)
        if not match:
            return prompt
        current = match.group(2).strip().rstrip(".")
        if blend_phrase.lower() in current.lower():
            return prompt
        updated = f"{match.group(1)}{current} {blend_phrase}."
        return re.sub(pattern, updated, prompt, count=1)
