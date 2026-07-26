"""Text-based genetic algorithm for prompt optimization.

This module evolves human-readable prompt variants using text mutations and
section-aware crossover. It relies on enriched prompt text and ML-selected
strategies to initialize the population, then iterates for a small number of
generations to produce the best optimized prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any

from app.services.domain_knowledge import get_domain_profile, infer_domain
from app.services.fitness import FitnessService


@dataclass(slots=True)
class PromptCandidate:
    """Represent a prompt candidate in the genetic search process."""

    prompt: str
    strategies: list[str]
    generation: int
    lineage: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize a candidate to a dictionary."""
        return {
            "prompt": self.prompt,
            "strategies": self.strategies,
            "generation": self.generation,
            "lineage": self.lineage,
        }


class GeneticAlgorithmService:
    """Manage population initialization, evolution, and final selection."""

    def __init__(
        self,
        fitness_service: FitnessService | None = None,
        population_size: int = 8,
        generations: int = 4,
        selection_method: str = "tournament",
        random_state: int = 42,
    ) -> None:
        """Initialize the GA service with configurable search parameters."""
        self.fitness_service = fitness_service or FitnessService()
        self.population_size = max(6, population_size)
        self.generations = min(max(3, generations), 5)
        self.selection_method = selection_method
        self.random = random.Random(random_state)

    def initialize_population(self, seed_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate an initial population from the enriched prompt and strategies."""
        base_prompt = str(
            seed_data.get("baseline_prompt")
            or seed_data.get("enriched_prompt")
            or seed_data.get("prompt")
            or ""
        ).strip()
        if not base_prompt:
            return []

        strategy_data = seed_data.get("strategy_data", {})
        selected_strategies = strategy_data.get("recommended_strategies", []) or ["direct_instruction", "role_based"]
        mode = str(seed_data.get("mode", "cost")).lower()
        domain_profile = self._resolve_domain_profile(seed_data, strategy_data, base_prompt)

        population: list[PromptCandidate] = [
            PromptCandidate(
                prompt=base_prompt,
                strategies=list(dict.fromkeys(selected_strategies)),
                generation=0,
                lineage="seed",
            )
        ]

        strategy_combinations = self._strategy_combinations(selected_strategies)
        for strategies in strategy_combinations:
            variant_prompt = self._apply_strategies(base_prompt, strategies, mode, domain_profile)
            population.append(
                PromptCandidate(
                    prompt=variant_prompt,
                    strategies=list(strategies),
                    generation=0,
                    lineage="strategy_init",
                )
            )

        while len(population) < self.population_size:
            sample_size = min(len(selected_strategies), self.random.randint(1, max(1, len(selected_strategies))))
            sampled = self.random.sample(selected_strategies, k=sample_size)
            variant = self._mutate_text(
                text=self._apply_strategies(base_prompt, sampled, mode, domain_profile),
                mode=mode,
                strategy_hints=selected_strategies,
                domain_profile=domain_profile,
            )
            population.append(
                PromptCandidate(
                    prompt=variant,
                    strategies=sampled,
                    generation=0,
                    lineage="strategy_mutation",
                )
            )

        return [candidate.to_dict() for candidate in self._deduplicate_candidates(population)]

    def evolve(
        self,
        population: list[dict[str, Any]],
        strategy_data: dict[str, Any],
        optimization_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the genetic algorithm and return the best prompt candidate."""
        if not population:
            return {"best_prompt": "", "best_candidate": None, "history": []}

        context = optimization_context or {}
        mode = str(context.get("mode", "cost")).lower()
        reference_prompt = str(context.get("reference_prompt") or context.get("prompt") or population[0]["prompt"]).strip()
        alpha = float(context.get("alpha", 0.55))
        beta = float(context.get("beta", 0.45))
        selected_strategies = strategy_data.get("recommended_strategies", []) or ["direct_instruction", "role_based"]
        domain_profile = self._resolve_domain_profile(context, strategy_data, reference_prompt)

        current_population = population
        history: list[dict[str, Any]] = []
        best_candidate: dict[str, Any] | None = None

        for generation in range(self.generations):
            scored_population = self.fitness_service.score_candidates(
                candidates=current_population,
                mode=mode,
                reference_prompt=reference_prompt,
                alpha=alpha,
                beta=beta,
            )

            current_best = scored_population[0]
            history.append(
                {
                    "generation": generation,
                    "best_fitness": current_best["fitness"],
                    "best_prompt": current_best["prompt"],
                    "strategies": current_best.get("strategies", []),
                }
            )

            if best_candidate is None or current_best["fitness"] > best_candidate["fitness"]:
                best_candidate = current_best

            if generation == self.generations - 1:
                break

            parents = self._select_parents(scored_population)
            offspring = self._reproduce(
                parents=parents,
                selected_strategies=selected_strategies,
                mode=mode,
                generation=generation + 1,
                domain_profile=domain_profile,
            )

            elite_count = min(2, len(scored_population))
            elites = [
                {
                    "prompt": candidate["prompt"],
                    "strategies": candidate.get("strategies", []),
                    "generation": generation + 1,
                    "lineage": "elite",
                }
                for candidate in scored_population[:elite_count]
            ]

            current_population = elites + offspring
            current_population = self._trim_population(current_population, self.population_size)

        return {
            "best_prompt": best_candidate["prompt"] if best_candidate else "",
            "best_candidate": best_candidate,
            "history": history,
            "mode": mode,
        }

    def optimize(self, seed_data: dict[str, Any], strategy_data: dict[str, Any]) -> dict[str, Any]:
        """Convenience method that initializes and evolves a full population."""
        initialization_seed = dict(seed_data)
        initialization_seed["strategy_data"] = strategy_data
        population = self.initialize_population(initialization_seed)
        return self.evolve(
            population=population,
            strategy_data=strategy_data,
            optimization_context={
                "mode": seed_data.get("mode", "cost"),
                "reference_prompt": seed_data.get("prompt") or seed_data.get("enriched_prompt", ""),
                "alpha": seed_data.get("alpha", 0.55),
                "beta": seed_data.get("beta", 0.45),
                "enrichment_metadata": seed_data.get("enrichment_metadata", {}),
                "domain": seed_data.get("domain") or seed_data.get("metadata", {}).get("domain"),
            },
        )

    def _strategy_combinations(self, strategies: list[str]) -> list[tuple[str, ...]]:
        """Build compact strategy combinations for the initial population."""
        combinations: list[tuple[str, ...]] = []
        for strategy in strategies:
            combinations.append((strategy,))
        if len(strategies) >= 2:
            combinations.append(tuple(strategies[:2]))
        if len(strategies) >= 3:
            combinations.append(tuple(strategies[:3]))
        return combinations

    def _apply_strategies(
        self,
        prompt: str,
        strategies: list[str] | tuple[str, ...],
        mode: str,
        domain_profile: dict[str, Any],
    ) -> str:
        """Apply selected strategies to seed prompt text."""
        transformed = prompt
        for strategy in strategies:
            transformed = self._apply_single_strategy(transformed, strategy, mode, domain_profile)
        return self._clean_prompt(transformed)

    def _apply_single_strategy(self, prompt: str, strategy: str, mode: str, domain_profile: dict[str, Any]) -> str:
        """Apply one strategy-specific text transformation."""
        if strategy == "direct_instruction":
            return self._ensure_section(prompt, "Task", self._extract_goal(prompt))
        if strategy == "compression":
            return self._compress_prompt(prompt)
        if strategy == "remove_examples":
            return self._remove_examples(prompt)
        if strategy == "minimal_output":
            return self._ensure_section(prompt, "Output", "Return only the essential answer.")
        if strategy == "role_based":
            return self._ensure_section(prompt, "Role", str(domain_profile.get("role", "Act as a domain expert.")))
        if strategy == "few_shot":
            return self._add_example(prompt, domain_profile)
        if strategy == "chain_of_thought":
            return self._ensure_section(prompt, "Reasoning", "Work through the task step by step before answering.")
        if strategy == "constraint_based":
            addition = f"Follow all constraints strictly and preserve the requested scope. {domain_profile.get('constraint', '')}".strip()
            return self._append_to_section(prompt, "Constraints", addition)
        return self._compress_prompt(prompt) if mode == "cost" else self._enrich_prompt(prompt, domain_profile)

    def _select_parents(self, scored_population: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Select parents using top-k or tournament selection."""
        if self.selection_method == "top_k":
            return scored_population[: max(2, min(4, len(scored_population)))]

        tournament_winners: list[dict[str, Any]] = []
        while len(tournament_winners) < max(2, min(4, len(scored_population))):
            competitors = self.random.sample(scored_population, k=min(3, len(scored_population)))
            tournament_winners.append(max(competitors, key=lambda item: item["fitness"]))
        return tournament_winners

    def _reproduce(
        self,
        parents: list[dict[str, Any]],
        selected_strategies: list[str],
        mode: str,
        generation: int,
        domain_profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Create offspring through crossover and mutation."""
        offspring: list[dict[str, Any]] = []
        while len(offspring) < self.population_size - 2:
            parent_a, parent_b = self.random.sample(parents, k=2) if len(parents) >= 2 else (parents[0], parents[0])
            child_prompt = self._crossover(parent_a["prompt"], parent_b["prompt"])
            child_prompt = self._mutate_text(
                child_prompt,
                mode=mode,
                strategy_hints=selected_strategies,
                domain_profile=domain_profile,
            )
            child_strategies = list(dict.fromkeys(parent_a.get("strategies", []) + parent_b.get("strategies", [])))
            offspring.append(
                {
                    "prompt": child_prompt,
                    "strategies": child_strategies[:4],
                    "generation": generation,
                    "lineage": "crossover_mutation",
                }
            )
        return offspring

    def _crossover(self, parent_a: str, parent_b: str) -> str:
        """Combine prompt sections from both parents into a readable child prompt."""
        sections_a = self._parse_sections(parent_a)
        sections_b = self._parse_sections(parent_b)
        ordered_sections = ["Role", "Task", "Goal", "Context", "Examples", "Constraints", "Tone", "Reasoning", "Output"]
        merged: list[str] = []

        for section in ordered_sections:
            if section == "Role" and sections_a.get("Role"):
                merged.append(f"Role: {sections_a['Role']}")
                continue
            if section == "Examples" and sections_b.get("Examples"):
                merged.append("Examples:")
                merged.extend(f"- {line}" for line in self._normalize_example_lines(sections_b["Examples"]))
                continue

            value = sections_a.get(section) or sections_b.get(section)
            if value:
                merged.append(f"{section}: {value}")

        if not merged:
            return self._clean_prompt(parent_a)
        return self._clean_prompt("\n".join(merged))

    def _mutate_text(self, text: str, mode: str, strategy_hints: list[str], domain_profile: dict[str, Any]) -> str:
        """Apply random text-level mutations while keeping prompts readable."""
        mutated = text
        mutation_pool = [
            lambda value: self._add_example(value, domain_profile),
            self._remove_examples,
            self._change_tone,
            self._simplify_wording,
            lambda value: self._inject_domain_vocabulary(value, domain_profile),
        ]

        mutation_count = self.random.randint(1, 2)
        for mutation in self.random.sample(mutation_pool, k=mutation_count):
            mutated = mutation(mutated)

        mutated = self._compress_prompt(mutated) if mode == "cost" else self._enrich_prompt(mutated, domain_profile)

        for strategy in self.random.sample(strategy_hints, k=min(len(strategy_hints), 1)):
            mutated = self._apply_single_strategy(mutated, strategy, mode, domain_profile)

        return self._clean_prompt(mutated)

    def _add_example(self, text: str, domain_profile: dict[str, Any]) -> str:
        """Add a short, human-readable example to the prompt."""
        examples = self._normalize_example_lines(self._parse_sections(text).get("Examples", ""))
        sample_examples = [
            "Input: rough request | Output: clear, precise instruction",
            "Input: vague question | Output: structured answer with constraints",
            "Input: short coding task | Output: direct expert-style prompt",
            str(domain_profile.get("example", "")).strip(),
        ]
        valid_examples = [example for example in sample_examples if example]
        candidate = self.random.choice(valid_examples)
        if candidate not in examples:
            examples.append(candidate)
        return self._set_examples(text, examples[:3])

    def _remove_examples(self, text: str) -> str:
        """Remove examples to reduce prompt cost and verbosity."""
        sections = self._parse_sections(text)
        if "Examples" not in sections:
            return text
        examples = self._normalize_example_lines(sections.get("Examples", ""))
        if len(examples) <= 1:
            sections.pop("Examples", None)
        else:
            examples.pop()
            sections["Examples"] = "\n".join(examples)
        return self._rebuild_sections(sections)

    def _change_tone(self, text: str) -> str:
        """Mutate the prompt tone while keeping it readable."""
        tone_options = [
            "professional and precise",
            "friendly and practical",
            "analytical and structured",
            "concise and directive",
        ]
        return self._ensure_section(text, "Tone", self.random.choice(tone_options))

    def _simplify_wording(self, text: str) -> str:
        """Replace verbose phrases with simpler wording."""
        replacements = {
            "in order to": "to",
            "with the objective of": "to",
            "it is important to note that": "",
            "please make sure to": "ensure",
            "provide a response that is": "be",
            "utilize": "use",
            "approximately": "about",
        }
        simplified = text
        for source, target in replacements.items():
            simplified = re.sub(source, target, simplified, flags=re.IGNORECASE)
        return self._clean_prompt(simplified)

    def _compress_prompt(self, text: str) -> str:
        """Apply cost-mode compression without destroying readability."""
        sections = self._parse_sections(text)
        compact_sections: list[str] = []
        for section in ["Role", "Task", "Goal", "Context", "Constraints", "Output", "Tone"]:
            value = sections.get(section)
            if not value:
                continue
            if section in {"Context", "Constraints"}:
                value = self._truncate_sentence(value, 18)
            compact_sections.append(f"{section}: {value}")
        if "Examples" in sections and self.random.random() < 0.3:
            compact_sections.append("Examples:\n- " + self._normalize_example_lines(sections["Examples"])[0])
        return self._clean_prompt("\n".join(compact_sections))

    def _enrich_prompt(self, text: str, domain_profile: dict[str, Any]) -> str:
        """Apply context-mode enrichment by adding helpful structure."""
        enriched = self._ensure_section(text, "Reasoning", "Consider the task carefully and keep the answer grounded.")
        enriched = self._append_to_section(enriched, "Constraints", "Preserve accuracy and relevant detail.")
        enriched = self._append_to_section(enriched, "Constraints", str(domain_profile.get("constraint", "")))
        enriched = self._ensure_section(
            enriched,
            "Context",
            self._build_domain_context(domain_profile),
        )
        enriched = self._ensure_section(enriched, "Output", "Return a well-structured answer that follows the task intent.")
        return self._clean_prompt(enriched)

    def _inject_domain_vocabulary(self, text: str, domain_profile: dict[str, Any]) -> str:
        """Blend related concepts into task and context sections."""
        blend_phrase = str(domain_profile.get("blend_phrase", "")).strip()
        guidance = str(domain_profile.get("guidance", "")).strip()
        if not blend_phrase and not guidance:
            return text
        updated = text
        if blend_phrase:
            updated = self._append_to_section(updated, "Task", blend_phrase)
        if guidance:
            updated = self._append_to_section(updated, "Context", guidance)
        return updated

    def _build_domain_context(self, domain_profile: dict[str, Any]) -> str:
        """Build a domain-aware context sentence for enriched prompts."""
        guidance = str(domain_profile.get("guidance", "")).strip()
        return guidance

    def _resolve_domain_profile(
        self,
        seed_data: dict[str, Any],
        strategy_data: dict[str, Any],
        prompt_text: str,
    ) -> dict[str, Any]:
        """Resolve a shared domain profile from seed data and ML features."""
        enrichment = dict(seed_data.get("enrichment_metadata", {}))
        features = dict(strategy_data.get("features", {}))
        domain = str(
            enrichment.get("domain")
            or seed_data.get("domain")
            or features.get("domain")
            or infer_domain(prompt_text)
        ).lower()
        profile = get_domain_profile(domain)
        profile["expanded_terms"] = list(
            enrichment.get("expanded_terms")
            or seed_data.get("related_terms")
            or seed_data.get("expanded_terms")
            or []
        )
        profile["blend_phrase"] = str(
            enrichment.get("blend_phrase")
            or seed_data.get("blend_phrase")
            or ""
        ).strip()
        return profile

    def _ensure_section(self, text: str, section: str, value: str) -> str:
        """Create or replace a named prompt section."""
        sections = self._parse_sections(text)
        sections[section] = value
        return self._rebuild_sections(sections)

    def _append_to_section(self, text: str, section: str, addition: str) -> str:
        """Append text to an existing section or create it if missing."""
        sections = self._parse_sections(text)
        current = sections.get(section, "")
        if addition not in current:
            sections[section] = f"{current} {addition}".strip() if current else addition
        return self._rebuild_sections(sections)

    def _set_examples(self, text: str, examples: list[str]) -> str:
        """Write normalized examples back into the prompt."""
        sections = self._parse_sections(text)
        sections["Examples"] = "\n".join(examples)
        return self._rebuild_sections(sections)

    def _extract_goal(self, text: str) -> str:
        """Extract a goal/task sentence from the prompt."""
        sections = self._parse_sections(text)
        return sections.get("Goal") or sections.get("Task") or self._truncate_sentence(text.replace("\n", " "), 18)

    def _parse_sections(self, text: str) -> dict[str, str]:
        """Parse section-based prompt text into a dictionary."""
        sections: dict[str, list[str]] = {}
        current_section = "Task"
        sections[current_section] = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^(Role|Task|Goal|Context|Examples|Constraints|Output|Tone|Reasoning):\s*(.*)$", stripped)
            if match:
                current_section = match.group(1)
                sections[current_section] = [match.group(2)] if match.group(2) else []
            else:
                sections.setdefault(current_section, []).append(stripped.lstrip("- ").strip())

        return {
            section: "\n".join(part for part in values if part).strip()
            for section, values in sections.items()
            if any(part for part in values)
        }

    def _rebuild_sections(self, sections: dict[str, str]) -> str:
        """Rebuild prompt text from section dictionary while preserving order."""
        ordered_sections = ["Role", "Task", "Goal", "Context", "Examples", "Constraints", "Tone", "Reasoning", "Output"]
        lines: list[str] = []
        for section in ordered_sections:
            value = sections.get(section)
            if not value:
                continue
            if section == "Examples":
                lines.append("Examples:")
                lines.extend(f"- {line}" for line in self._normalize_example_lines(value))
            else:
                lines.append(f"{section}: {value}")
        return self._clean_prompt("\n".join(lines))

    def _normalize_example_lines(self, value: str) -> list[str]:
        """Normalize example lines into a compact list."""
        return [line.strip().lstrip("- ").strip() for line in value.splitlines() if line.strip()]

    def _truncate_sentence(self, text: str, max_words: int) -> str:
        """Trim text to a maximum number of words."""
        words = text.split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]).rstrip(",;:") + "."

    def _trim_population(self, population: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
        """Limit population size while removing duplicates."""
        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in population:
            prompt = candidate.get("prompt", "")
            if prompt in seen:
                continue
            seen.add(prompt)
            deduplicated.append(candidate)
            if len(deduplicated) >= size:
                break
        return deduplicated

    def _deduplicate_candidates(self, candidates: list[PromptCandidate]) -> list[PromptCandidate]:
        """Remove duplicate prompts from an initial candidate list."""
        deduplicated: list[PromptCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized_prompt = candidate.prompt.strip()
            if normalized_prompt in seen:
                continue
            seen.add(normalized_prompt)
            deduplicated.append(candidate)
        return deduplicated[: self.population_size]

    def _clean_prompt(self, text: str) -> str:
        """Normalize spacing and keep prompt text readable."""
        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
