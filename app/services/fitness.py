"""Text-based fitness scoring for prompt optimization candidates.

This module evaluates prompt candidates using lightweight heuristic metrics that
operate on plain text only. The scoring logic supports both cost-focused and
context-focused optimization modes while preserving prompt readability.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class CandidateMetrics:
    """Metrics collected for a single prompt candidate."""

    token_count: int
    output_quality: float
    semantic_richness: float
    accuracy: float
    fitness: float

    def to_dict(self) -> dict[str, float]:
        """Serialize candidate metrics for API consumers."""
        return {
            "token_count": float(self.token_count),
            "output_quality": round(self.output_quality, 4),
            "semantic_richness": round(self.semantic_richness, 4),
            "accuracy": round(self.accuracy, 4),
            "fitness": round(self.fitness, 4),
        }


class FitnessService:
    """Score human-readable prompt candidates in cost or context mode."""

    _PROMPT_STRUCTURE_MARKERS = {
        "Role:",
        "Task:",
        "Goal:",
        "Examples:",
        "Constraints:",
        "Output:",
        "Tone:",
        "Reasoning:",
        "Context:",
    }

    def score_candidates(
        self,
        candidates: list[dict[str, Any]],
        mode: str,
        reference_prompt: str,
        alpha: float = 0.55,
        beta: float = 0.45,
    ) -> list[dict[str, Any]]:
        """Assign fitness scores to candidates and return them sorted by quality."""
        scored_candidates = [
            self.score_candidate(
                candidate=candidate,
                mode=mode,
                reference_prompt=reference_prompt,
                alpha=alpha,
                beta=beta,
            )
            for candidate in candidates
        ]
        return sorted(scored_candidates, key=lambda item: item["fitness"], reverse=True)

    def score_candidate(
        self,
        candidate: dict[str, Any],
        mode: str,
        reference_prompt: str,
        alpha: float = 0.55,
        beta: float = 0.45,
    ) -> dict[str, Any]:
        """Score a single candidate prompt using the selected optimization mode."""
        text = str(candidate.get("prompt", "")).strip()
        token_count = self.token_count(text)
        output_quality = self.output_quality_score(text)
        semantic_richness = self.semantic_richness_score(text)
        accuracy = self.accuracy_score(reference_prompt, text)

        if mode == "context":
            fitness = alpha * semantic_richness + beta * accuracy
        else:
            inverse_token_count = 1.0 / max(token_count, 1)
            fitness = alpha * inverse_token_count + beta * output_quality

        metrics = CandidateMetrics(
            token_count=token_count,
            output_quality=output_quality,
            semantic_richness=semantic_richness,
            accuracy=accuracy,
            fitness=fitness,
        )

        enriched_candidate = dict(candidate)
        enriched_candidate["fitness"] = round(metrics.fitness, 4)
        enriched_candidate["metrics"] = metrics.to_dict()
        return enriched_candidate

    def token_count(self, text: str) -> int:
        """Count whitespace-delimited text tokens."""
        return len(re.findall(r"\b\w+\b", text))

    def semantic_similarity(self, source_text: str, candidate_text: str) -> float:
        """Compute semantic similarity between two prompts using TF-IDF cosine similarity."""
        source = source_text.strip()
        candidate = candidate_text.strip()

        if not source and not candidate:
            return 1.0
        if not source or not candidate:
            return 0.0

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([source, candidate])
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(similarity), 4)

    def output_quality_score(self, text: str) -> float:
        """Estimate readability and directness for cost-focused optimization."""
        token_count = self.token_count(text)
        sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
        avg_sentence_length = token_count / max(len(sentences), 1)
        structure_score = min(
            sum(0.08 for marker in self._PROMPT_STRUCTURE_MARKERS if marker in text),
            0.32,
        )
        verbosity_penalty = min(max(token_count - 140, 0) / 260, 0.35)
        sentence_penalty = min(max(avg_sentence_length - 20, 0) / 50, 0.2)
        clarity_bonus = 0.2 if "Task:" in text or "Goal:" in text else 0.0
        action_bonus = 0.1 if re.search(r"\b(write|solve|explain|generate|optimize|summarize)\b", text.lower()) else 0.0

        raw = 0.45 + structure_score + clarity_bonus + action_bonus - verbosity_penalty - sentence_penalty
        return round(max(0.0, min(raw, 1.0)), 4)

    def semantic_richness_score(self, text: str) -> float:
        """Estimate how much useful contextual information the prompt carries."""
        marker_hits = sum(1 for marker in self._PROMPT_STRUCTURE_MARKERS if marker in text)
        unique_tokens = {token.lower() for token in re.findall(r"\b\w+\b", text)}
        lexical_diversity = min(len(unique_tokens) / 80, 0.35)
        example_bonus = 0.15 if "Examples:" in text else 0.0
        reasoning_bonus = 0.15 if "Reasoning:" in text else 0.0
        constraint_bonus = 0.15 if "Constraints:" in text else 0.0
        role_bonus = 0.1 if "Role:" in text else 0.0
        base = min(marker_hits * 0.07, 0.4) + lexical_diversity + example_bonus + reasoning_bonus + constraint_bonus + role_bonus
        return round(max(0.0, min(base, 1.0)), 4)

    def accuracy_score(self, reference_prompt: str, candidate_text: str) -> float:
        """Estimate instruction preservation via keyword overlap with the source."""
        reference_terms = self._important_terms(reference_prompt)
        candidate_terms = self._important_terms(candidate_text)

        if not reference_terms:
            return 0.6

        overlap = len(reference_terms & candidate_terms) / len(reference_terms)
        directive_bonus = 0.1 if any(marker in candidate_text for marker in {"Task:", "Goal:", "Constraints:"}) else 0.0
        return round(max(0.0, min(overlap + directive_bonus, 1.0)), 4)

    def improvement_score(
        self,
        original_prompt: str,
        optimized_prompt: str,
        mode: str = "cost",
        semantic_weight: float = 0.5,
        quality_weight: float = 0.5,
    ) -> float:
        """Compute a blended improvement score for an optimized prompt."""
        semantic_score = self.semantic_similarity(original_prompt, optimized_prompt)
        optimized_quality = self.output_quality_score(optimized_prompt)
        original_quality = self.output_quality_score(original_prompt)
        quality_delta = max(optimized_quality - original_quality, 0.0)
        token_before = self.token_count(original_prompt)
        token_after = self.token_count(optimized_prompt)
        compression_gain = max(token_before - token_after, 0) / max(token_before, 1)
        richness_gain = max(
            self.semantic_richness_score(optimized_prompt) - self.semantic_richness_score(original_prompt),
            0.0,
        )

        mode_gain = compression_gain if mode == "cost" else richness_gain
        score = semantic_weight * semantic_score + quality_weight * min(quality_delta + mode_gain, 1.0)
        return round(max(0.0, min(score, 1.0)), 4)

    def _important_terms(self, text: str) -> set[str]:
        """Extract informative terms while removing generic stopwords."""
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "for",
            "with",
            "of",
            "on",
            "in",
            "this",
            "that",
            "is",
            "are",
            "be",
            "as",
            "by",
            "it",
            "from",
            "use",
        }
        return {
            token.lower()
            for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", text)
            if token.lower() not in stopwords
        }
