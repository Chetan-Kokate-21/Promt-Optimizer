"""Final formatting utilities for polished optimized prompts."""

from __future__ import annotations

import re
from typing import Any

from app.services.domain_knowledge import get_domain_profile


class PromptFormatterService:
    """Normalize optimized prompts into a consistent presentation template."""

    _ORDER = ["Role", "Task", "Context", "Constraints", "Output", "Example"]

    def format_prompt(
        self,
        prompt: str,
        mode: str,
        domain: str,
        domain_term_expansion: dict[str, Any] | None = None,
    ) -> str:
        """Return a polished prompt with consistent section formatting."""
        text = str(prompt or "").strip()
        if not text:
            return ""

        if mode != "context":
            return self._clean(text)

        sections = self._parse_sections(text)
        profile = get_domain_profile(domain)
        blend_phrase = str((domain_term_expansion or {}).get("blend_phrase", "")).strip()

        role = sections.get("Role") or str(profile.get("role", "Act as a helpful expert.")).strip()
        task = self._normalize_task(sections.get("Task") or sections.get("Goal") or text, blend_phrase)
        context = self._normalize_context(sections.get("Context") or str(profile.get("guidance", "")).strip())
        constraints = self._normalize_bullets(
            sections.get("Constraints"),
            fallback_items=self._default_constraint_items(profile, blend_phrase),
        )
        output = self._normalize_bullets(
            sections.get("Output"),
            fallback_items=self._default_output_items(domain),
        )
        example = self._normalize_example(sections.get("Example") or sections.get("Examples") or str(profile.get("example", "")).strip())

        rendered = [
            ("Role", role),
            ("Task", task),
            ("Context", context),
            ("Constraints", constraints),
            ("Output", output),
            ("Example", example),
        ]
        return "\n\n".join(f"{name}: {value}" if name in {"Role", "Task", "Context"} else f"{name}:\n{value}" for name, value in rendered if value).strip()

    def _parse_sections(self, text: str) -> dict[str, str]:
        """Parse loose structured text into prompt sections."""
        sections: dict[str, list[str]] = {}
        current: str | None = None
        seen_labeled_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(Role|Task|Goal|Context|Constraints|Output|Example|Examples|Tone|Reasoning):\s*(.*)$", line)
            if match:
                current = match.group(1)
                seen_labeled_section = True
                sections[current] = [match.group(2)] if match.group(2) else []
            else:
                if not seen_labeled_section:
                    current = "Task"
                    sections.setdefault(current, []).append(line.lstrip("- ").strip())
                    continue
                if current in {"Role", "Context"}:
                    continue
                if current is None:
                    current = "Task"
                sections.setdefault(current, []).append(line.lstrip("- ").strip())

        return {
            section: "\n".join(part for part in values if part).strip()
            for section, values in sections.items()
            if any(part for part in values)
        }

    def _normalize_task(self, task: str, blend_phrase: str) -> str:
        """Ensure the task reads like a clear instruction."""
        cleaned = self._clean(task).rstrip(".")
        cleaned = re.sub(r"^(give|provide)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned[:1].upper() + cleaned[1:] if cleaned else "Provide a clear response"
        if blend_phrase and blend_phrase.lower() not in cleaned.lower():
            cleaned = f"{cleaned} {blend_phrase}"
        return cleaned.rstrip(".") + "."

    def _normalize_context(self, context: str) -> str:
        """Turn context into smooth prose instead of keyword dumps."""
        cleaned = self._clean(context).strip()
        cleaned = re.sub(r"^(Domain=.*?;\s*)?(Intent=.*?;\s*)?(Audience=.*?;\s*)?", "", cleaned, count=1)
        cleaned = re.sub(r"^Focus on terms like .*?\.\s*", "", cleaned)
        cleaned = re.sub(r"^Add related terms such as .*?\.\s*", "", cleaned)
        cleaned = re.sub(r"^Focus terms: .*?\.\s*", "", cleaned)
        cleaned = re.sub(r"^Related vocabulary: .*?\.\s*", "", cleaned)
        return cleaned.rstrip(".") + "." if cleaned else "Preserve the relevant domain context, key details, and technical terminology."

    def _normalize_bullets(self, content: str | None, fallback_items: list[str]) -> str:
        """Render structured bullets from freeform or newline content."""
        items = self._extract_items(content) if content else []
        if not items:
            items = fallback_items
        cleaned_items = []
        for item in items:
            normalized = self._clean(item).strip().rstrip(".")
            if normalized:
                cleaned_items.append(f"- {normalized}")
        return "\n".join(cleaned_items)

    def _extract_items(self, content: str) -> list[str]:
        """Split content into bullet-like items."""
        text = str(content or "").strip()
        if not text:
            return []
        lines = [line.strip().lstrip("- ").strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
        sentence_parts = [part.strip() for part in re.split(r"\.\s+|;\s+", text) if part.strip()]
        return sentence_parts or [text]

    def _default_constraint_items(self, profile: dict[str, Any], blend_phrase: str) -> list[str]:
        """Return fallback constraint bullets."""
        items = [
            "Provide concrete implementation details when relevant",
            str(profile.get("constraint", "Preserve accuracy and relevant detail")).strip().rstrip("."),
        ]
        if blend_phrase:
            items.append(f"Blend related technical context naturally {blend_phrase.replace('while covering', 'by covering')}".rstrip("."))
        return items

    def _default_output_items(self, domain: str) -> list[str]:
        """Return fallback output bullets, lightly tailored by domain."""
        items = [
            "Use structured sections",
            "Highlight practical trade-offs and key decisions",
        ]
        if domain in {"coding", "data"}:
            items.insert(1, "Include code snippets or implementation-oriented guidance when useful")
        elif domain == "math":
            items.insert(1, "Show worked steps and intermediate reasoning where relevant")
        return items

    def _normalize_example(self, example: str) -> str:
        """Ensure example section uses the desired inline format."""
        cleaned = self._clean(example)
        if not cleaned:
            return ""
        return cleaned

    def _clean(self, text: str) -> str:
        """Normalize whitespace."""
        normalized = re.sub(r"[ \t]+", " ", text or "")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()
