"""Prompt enrichment utilities for preparing GA-ready seed prompts.

This module builds a richer prompt representation from the raw request by
organizing instructions, context, constraints, and metadata into readable text
sections that can be mutated and recombined by the genetic algorithm.
"""

from __future__ import annotations

from typing import Any

from app.services.domain_knowledge import get_domain_profile, infer_domain


class EnrichmentService:
    """Create structured, human-readable enriched prompt text."""

    def enrich(self, prompt_data: dict[str, Any]) -> dict[str, Any]:
        """Construct an enriched prompt artifact from the incoming payload."""
        prompt = str(prompt_data.get("prompt", "")).strip()
        context = prompt_data.get("context", {})
        constraints = prompt_data.get("constraints", {})
        metadata = prompt_data.get("metadata", {})
        mode = str(prompt_data.get("mode", metadata.get("mode", "cost"))).lower()

        domain = self._value(metadata, "domain", default=infer_domain(prompt, metadata))
        domain_profile = get_domain_profile(domain)
        expanded_terms = [str(term).strip() for term in metadata.get("domain_terms", []) if str(term).strip()]
        blend_phrase = str(prompt_data.get("domain_term_expansion", {}).get("blend_phrase", "")).strip()
        role = self._value(context, "role", default=str(domain_profile["role"]))
        intent = self._value(metadata, "intent", default="optimize")
        audience = self._value(context, "audience", default="end user")
        tone = self._value(context, "tone", default="clear and professional")
        examples = context.get("examples", [])
        if isinstance(examples, str):
            examples = [examples]
        if mode == "context":
            domain_example = str(domain_profile["example"])
            if domain_example and domain_example not in examples:
                examples = [*examples, domain_example]

        enriched_prompt = "\n".join(
            segment
            for segment in [
                f"Role: {role}",
                self._render_goal(prompt, blend_phrase),
                self._render_context(domain, intent, audience, domain_profile, mode),
                f"Tone: {tone}",
                self._render_examples(examples),
                self._render_constraints(constraints, mode, domain_profile, blend_phrase),
            ]
            if segment
        )

        enriched_data = dict(prompt_data)
        enriched_data["mode"] = mode if mode in {"cost", "context"} else "cost"
        enriched_data["enriched_prompt"] = enriched_prompt
        enriched_data["enrichment_metadata"] = {
            "domain": domain,
            "intent": intent,
            "audience": audience,
            "tone": tone,
            "example_count": len(examples),
            "domain_keywords": list(domain_profile["keywords"]),
            "expanded_terms": expanded_terms,
            "blend_phrase": blend_phrase,
            "domain_guidance": str(domain_profile["guidance"]),
            "domain_constraint": str(domain_profile["constraint"]),
            "domain_example": str(domain_profile["example"]),
            "domain_role": str(domain_profile["role"]),
        }
        return enriched_data

    def _render_goal(self, prompt: str, blend_phrase: str) -> str:
        """Blend added technical context into the goal instead of listing terms separately."""
        if not prompt:
            return "Goal: Improve the supplied instruction"
        suffix = f" {blend_phrase}" if blend_phrase else ""
        return f"Goal: {prompt.rstrip('.')} {suffix}".strip() + "."

    def _render_examples(self, examples: list[str]) -> str:
        """Format examples as a readable prompt section."""
        if not examples:
            return ""
        lines = ["Examples:"]
        lines.extend(f"- {example.strip()}" for example in examples if example and example.strip())
        return "\n".join(lines)

    def _render_context(
        self,
        domain: str,
        intent: str,
        audience: str,
        domain_profile: dict[str, Any],
        mode: str,
    ) -> str:
        """Render the context section with domain guidance only."""
        base = f"Context: Domain={domain}; Intent={intent}; Audience={audience}"
        if mode != "context":
            return base
        guidance = str(domain_profile.get("guidance", "")).strip()
        detail_parts = [guidance] if guidance else []
        return f"{base}; {' '.join(detail_parts)}".strip()

    def _render_constraints(
        self,
        constraints: dict[str, Any],
        mode: str,
        domain_profile: dict[str, Any],
        blend_phrase: str,
    ) -> str:
        """Format constraint information and include mode guidance."""
        rendered = ["Constraints:"]
        if constraints:
            rendered.extend(f"- {key}: {value}" for key, value in constraints.items())
        else:
            rendered.append("- Preserve clarity and keep the prompt human-readable.")

        if mode == "cost":
            rendered.append("- Prefer concise wording and avoid unnecessary verbosity.")
        else:
            rendered.append("- Add enough context for accurate, grounded responses.")
            rendered.append(f"- {domain_profile.get('constraint', 'Preserve domain-specific accuracy and detail.')}")
            if blend_phrase:
                rendered.append(f"- Enrich the request naturally {blend_phrase.replace('while covering', 'by covering')}.")
        return "\n".join(rendered)

    def _value(self, data: dict[str, Any], key: str, default: str) -> str:
        """Fetch a value from a dictionary with a string fallback."""
        value = data.get(key, default)
        return str(value).strip() if value is not None else default
