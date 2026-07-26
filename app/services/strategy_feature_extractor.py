"""Feature extraction utilities for ML-based prompt strategy prediction.

This module converts raw prompt payloads into structured features that can be
consumed by classical machine learning models. The extracted features focus on
signals that influence strategy choice, including prompt size, difficulty,
ambiguity, domain classification, intent classification, and whether the task
appears to require multi-step reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


DOMAIN_CODING = "coding"
DOMAIN_MATH = "math"
DOMAIN_GENERAL = "general"

INTENT_GENERATE = "generate"
INTENT_EXPLAIN = "explain"
INTENT_SUMMARIZE = "summarize"
INTENT_SOLVE = "solve"
INTENT_ANALYZE = "analyze"
INTENT_OPTIMIZE = "optimize"
INTENT_CLASSIFY = "classify"
INTENT_OTHER = "other"

SUPPORTED_DOMAINS = [DOMAIN_CODING, DOMAIN_MATH, DOMAIN_GENERAL]
SUPPORTED_INTENTS = [
    INTENT_GENERATE,
    INTENT_EXPLAIN,
    INTENT_SUMMARIZE,
    INTENT_SOLVE,
    INTENT_ANALYZE,
    INTENT_OPTIMIZE,
    INTENT_CLASSIFY,
    INTENT_OTHER,
]


@dataclass(slots=True)
class PromptFeatures:
    """Structured feature set used by the strategy prediction model."""

    token_length: int
    complexity_score: float
    ambiguity_score: float
    domain: str
    intent_type: str
    requires_reasoning: bool

    def to_model_vector(self) -> list[float]:
        """Convert the feature object into a numeric model-ready vector."""
        domain_flags = [1.0 if self.domain == domain else 0.0 for domain in SUPPORTED_DOMAINS]
        intent_flags = [1.0 if self.intent_type == intent else 0.0 for intent in SUPPORTED_INTENTS]

        return [
            float(self.token_length),
            float(self.complexity_score),
            float(self.ambiguity_score),
            1.0 if self.requires_reasoning else 0.0,
            *domain_flags,
            *intent_flags,
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the extracted features into a plain dictionary."""
        return {
            "token_length": self.token_length,
            "complexity_score": self.complexity_score,
            "ambiguity_score": self.ambiguity_score,
            "domain": self.domain,
            "intent_type": self.intent_type,
            "requires_reasoning": self.requires_reasoning,
        }


class StrategyFeatureExtractor:
    """Extract domain-aware prompt features for strategy selection."""

    _CODING_KEYWORDS = {
        "code",
        "python",
        "java",
        "javascript",
        "bug",
        "debug",
        "api",
        "function",
        "class",
        "flask",
        "sql",
        "algorithm",
        "refactor",
        "compile",
    }
    _MATH_KEYWORDS = {
        "math",
        "equation",
        "solve",
        "integral",
        "derivative",
        "algebra",
        "geometry",
        "probability",
        "matrix",
        "calculate",
        "theorem",
        "proof",
    }
    _AMBIGUOUS_TERMS = {
        "maybe",
        "somehow",
        "possibly",
        "kind of",
        "sort of",
        "etc",
        "thing",
        "stuff",
        "something",
        "anything",
        "whatever",
        "around",
    }
    _REASONING_TERMS = {
        "why",
        "reason",
        "analyze",
        "compare",
        "tradeoff",
        "derive",
        "prove",
        "debug",
        "step-by-step",
        "step by step",
        "think through",
        "optimize",
    }
    _INTENT_KEYWORDS = {
        INTENT_GENERATE: {"write", "create", "draft", "generate", "produce", "design", "build"},
        INTENT_EXPLAIN: {"explain", "describe", "teach", "clarify"},
        INTENT_SUMMARIZE: {"summarize", "summary", "condense", "compress"},
        INTENT_SOLVE: {"solve", "compute", "calculate", "fix", "debug"},
        INTENT_ANALYZE: {"analyze", "evaluate", "review", "compare", "assess"},
        INTENT_OPTIMIZE: {"optimize", "improve", "enhance", "refine"},
        INTENT_CLASSIFY: {"classify", "categorize", "label", "detect"},
    }

    def extract(self, prompt_data: dict[str, Any]) -> PromptFeatures:
        """Extract predictive features from a prompt optimization payload."""
        prompt = self._build_text(prompt_data)
        tokens = self._tokenize(prompt)

        token_length = len(tokens)
        complexity_score = self._compute_complexity_score(prompt, tokens)
        ambiguity_score = self._compute_ambiguity_score(prompt, tokens)
        domain = self._detect_domain(prompt)
        intent_type = self._detect_intent(prompt)
        requires_reasoning = self._detect_reasoning(prompt, complexity_score, domain)

        return PromptFeatures(
            token_length=token_length,
            complexity_score=complexity_score,
            ambiguity_score=ambiguity_score,
            domain=domain,
            intent_type=intent_type,
            requires_reasoning=requires_reasoning,
        )

    def _build_text(self, prompt_data: dict[str, Any]) -> str:
        """Flatten prompt and auxiliary payload fields into one text blob."""
        prompt = str(prompt_data.get("prompt", "")).strip()
        context = prompt_data.get("context", {})
        constraints = prompt_data.get("constraints", {})
        metadata = prompt_data.get("metadata", {})

        parts = [prompt]
        for extra in (context, constraints, metadata):
            if isinstance(extra, dict):
                parts.extend(str(value) for value in extra.values() if value is not None)
            elif extra:
                parts.append(str(extra))
        return " ".join(part for part in parts if part)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lower-cased word units."""
        return re.findall(r"\b\w+\b", text.lower())

    def _compute_complexity_score(self, text: str, tokens: list[str]) -> float:
        """Estimate prompt complexity from lexical and structural signals."""
        if not tokens:
            return 0.0

        average_token_length = sum(len(token) for token in tokens) / len(tokens)
        long_token_ratio = sum(1 for token in tokens if len(token) >= 8) / len(tokens)
        punctuation_density = sum(text.count(mark) for mark in [",", ";", ":", "(", ")"]) / max(len(tokens), 1)
        structural_markers = sum(
            text.lower().count(marker)
            for marker in ["if", "while", "unless", "because", "compare", "analyze", "tradeoff"]
        )

        raw_score = (
            average_token_length * 0.18
            + long_token_ratio * 3.4
            + punctuation_density * 1.6
            + structural_markers * 0.35
        )
        return round(min(raw_score, 10.0), 3)

    def _compute_ambiguity_score(self, text: str, tokens: list[str]) -> float:
        """Estimate ambiguity from vague wording and missing specificity."""
        if not tokens:
            return 0.0

        lowered = text.lower()
        ambiguous_hits = sum(lowered.count(term) for term in self._AMBIGUOUS_TERMS)
        pronoun_hits = sum(1 for token in tokens if token in {"it", "this", "that", "they", "them"})
        question_marks = text.count("?")
        specificity_bonus = 1 if re.search(r"\b\d+\b", lowered) else 0

        raw_score = ambiguous_hits * 1.4 + pronoun_hits * 0.25 + question_marks * 0.2 - specificity_bonus * 0.6
        return round(max(0.0, min(raw_score, 10.0)), 3)

    def _detect_domain(self, text: str) -> str:
        """Classify prompt domain as coding, math, or general."""
        tokens = set(self._tokenize(text))
        coding_hits = len(tokens & self._CODING_KEYWORDS)
        math_hits = len(tokens & self._MATH_KEYWORDS)

        if coding_hits > math_hits and coding_hits > 0:
            return DOMAIN_CODING
        if math_hits > 0:
            return DOMAIN_MATH
        return DOMAIN_GENERAL

    def _detect_intent(self, text: str) -> str:
        """Classify the dominant prompt intent using keyword voting."""
        tokens = set(self._tokenize(text))
        best_intent = INTENT_OTHER
        best_score = 0

        for intent, keywords in self._INTENT_KEYWORDS.items():
            score = len(tokens & keywords)
            if score > best_score:
                best_intent = intent
                best_score = score

        return best_intent

    def _detect_reasoning(self, text: str, complexity_score: float, domain: str) -> bool:
        """Decide whether the prompt likely needs non-trivial reasoning."""
        lowered = text.lower()
        term_match = any(term in lowered for term in self._REASONING_TERMS)
        structural_cues = any(marker in lowered for marker in ["step", "multi", "constraint", "justify"])
        domain_cue = domain in {DOMAIN_CODING, DOMAIN_MATH}
        return bool(term_match or structural_cues or complexity_score >= 4.5 or domain_cue)
