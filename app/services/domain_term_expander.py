"""Semantic expansion of context-carrying technical terms for prompt enrichment."""

from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.domain_knowledge import get_domain_profile, infer_domain

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover - fallback path
    SentenceTransformer = None
    util = None


DOMAIN_TERM_BANK: dict[str, list[str]] = {
    "data": [
        "data pipeline",
        "etl orchestration",
        "data warehouse",
        "data lake",
        "analytics sandbox",
        "staging layer",
        "schema evolution",
        "schema drift",
        "data lineage",
        "data quality checks",
        "batch processing",
        "incremental loads",
        "data validation",
        "transformation logic",
        "governance policies",
        "observability metrics",
        "table partitioning",
        "query performance",
        "dbt models",
        "airflow dags",
    ],
    "coding": [
        "root cause analysis",
        "failure mode",
        "implementation detail",
        "api contract",
        "edge case handling",
        "input validation",
        "runtime behavior",
        "error propagation",
        "test coverage",
        "performance bottleneck",
        "state management",
        "integration point",
    ],
    "math": [
        "variable definition",
        "intermediate derivation",
        "symbolic manipulation",
        "substitution step",
        "final verification",
        "closed-form result",
        "boundary condition",
        "proof structure",
        "formula interpretation",
    ],
    "finance": [
        "cash flow assumptions",
        "revenue forecast",
        "cost structure",
        "sensitivity analysis",
        "risk exposure",
        "unit economics",
        "margin impact",
        "capital allocation",
        "budget variance",
    ],
    "healthcare": [
        "clinical context",
        "risk factor assessment",
        "differential considerations",
        "evidence-based guidance",
        "safety precautions",
        "red flag symptoms",
        "patient history",
    ],
    "legal": [
        "scope limitation",
        "jurisdictional nuance",
        "compliance risk",
        "contractual obligation",
        "liability exposure",
        "clause interpretation",
        "statutory context",
    ],
    "writing": [
        "audience intent",
        "narrative structure",
        "tone consistency",
        "revision goal",
        "message clarity",
        "voice alignment",
        "draft cohesion",
    ],
    "science": [
        "causal mechanism",
        "supporting evidence",
        "experimental setup",
        "variable control",
        "result interpretation",
        "method limitation",
        "testable hypothesis",
    ],
    "general": [
        "task context",
        "scope definition",
        "relevant constraints",
        "structured output",
        "useful examples",
        "key assumptions",
    ],
}

GLOBAL_TECHNICAL_TERMS: list[str] = [
    "workflow dependencies",
    "failure scenarios",
    "operational constraints",
    "system boundaries",
    "validation criteria",
    "implementation tradeoffs",
    "performance considerations",
    "monitoring signals",
    "integration points",
    "exception handling",
    "quality checks",
    "environment assumptions",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "into",
    "is", "it", "me", "of", "on", "or", "please", "show", "the", "this", "to", "use", "want", "with",
}


class DomainTermExpansionService:
    """Expand prompts with semantically related domain-specific vocabulary."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the expander with optional embedding support."""
        self.embedding_model_name = embedding_model_name
        self._embedding_model: Any | None = None

    def expand(self, prompt_data: dict[str, Any], top_k: int = 6) -> dict[str, Any]:
        """Attach domain-specific related terms to the prompt metadata."""
        prompt = str(prompt_data.get("prompt", "")).strip()
        metadata = dict(prompt_data.get("metadata", {}))
        domain = infer_domain(prompt, metadata)
        profile = get_domain_profile(domain)
        seed_terms = self._extract_seed_terms(prompt, profile)
        candidates = self._candidate_terms(domain, profile)
        related_terms = self._rank_related_terms(prompt, seed_terms, candidates, top_k=top_k)

        enriched = dict(prompt_data)
        metadata.update(
            {
                "domain": domain,
                "domain_terms": related_terms,
                "domain_seed_terms": seed_terms,
            }
        )
        enriched["metadata"] = metadata
        enriched["domain_term_expansion"] = {
            "domain": domain,
            "seed_terms": seed_terms,
            "related_terms": related_terms,
            "blend_phrase": self._build_blend_phrase(related_terms),
            "term_source": "sentence-transformers" if self._embedding_model_available() else "tfidf-fallback",
        }
        return enriched

    def _extract_seed_terms(self, prompt: str, profile: dict[str, Any]) -> list[str]:
        """Pull the most context-carrying phrases from the prompt itself."""
        lowered = prompt.lower()
        seeds: list[str] = []

        acronyms = re.findall(r"\b[A-Z]{2,}\b", prompt)
        seeds.extend(acronyms)

        phrase_matches = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*(?:\s+[a-zA-Z][a-zA-Z0-9_-]*){0,2}\b", prompt)
        scored_phrases = sorted(
            ((phrase, self._phrase_score(phrase, lowered, profile)) for phrase in set(phrase_matches)),
            key=lambda item: item[1],
            reverse=True,
        )
        for phrase, score in scored_phrases[:6]:
            cleaned = phrase.strip()
            if score > 0 and cleaned:
                seeds.append(cleaned)

        if not seeds and prompt:
            seeds.append(prompt)
        return list(dict.fromkeys(seed.strip() for seed in seeds if seed.strip()))

    def _candidate_terms(self, domain: str, profile: dict[str, Any]) -> list[str]:
        """Build a candidate pool from static bank and domain profile keywords."""
        bank_terms = DOMAIN_TERM_BANK.get(domain, []) + DOMAIN_TERM_BANK["general"] + GLOBAL_TECHNICAL_TERMS
        profile_terms = [str(term) for term in profile.get("keywords", [])]
        return list(dict.fromkeys([*profile_terms, *bank_terms]))

    def _rank_related_terms(
        self,
        prompt: str,
        seed_terms: list[str],
        candidates: list[str],
        top_k: int,
    ) -> list[str]:
        """Rank candidate domain terms with embeddings or TF-IDF fallback."""
        query = " ; ".join([prompt, *seed_terms]).strip(" ;")
        if not query or not candidates:
            return []

        if self._embedding_model_available():
            model = self._load_embedding_model()
            query_embedding = model.encode(query, convert_to_tensor=True)
            candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
            similarity_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
            ranked_indices = similarity_scores.argsort(descending=True).tolist()
            ordered = [candidates[index] for index in ranked_indices]
            return self._filter_terms(query, ordered, top_k=top_k)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([query, *candidates])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        ranked_pairs = sorted(zip(candidates, similarities), key=lambda item: item[1], reverse=True)
        ordered = [term for term, _ in ranked_pairs]
        return self._filter_terms(query, ordered, top_k=top_k)

    def _filter_terms(self, query: str, candidates: list[str], top_k: int) -> list[str]:
        """Drop duplicates and phrases already present in the query text."""
        lowered_query = query.lower()
        selected: list[str] = []
        for term in candidates:
            cleaned = term.strip()
            if (
                not cleaned
                or cleaned.lower() in lowered_query
                or self._is_too_close_to_query(cleaned, lowered_query)
            ):
                continue
            selected.append(cleaned)
            if len(selected) >= top_k:
                break
        return selected

    def _phrase_score(self, phrase: str, lowered_prompt: str, profile: dict[str, Any]) -> float:
        """Score phrases by specificity so the prompt drives expansion choices."""
        tokens = [token for token in re.findall(r"\b\w+\b", phrase.lower()) if token not in STOPWORDS]
        if not tokens:
            return 0.0
        unique_count = len(set(tokens))
        keyword_overlap = len(set(tokens) & {str(item).lower() for item in profile.get("keywords", [])})
        acronym_bonus = 0.8 if phrase.isupper() and len(phrase) > 1 else 0.0
        multiword_bonus = 0.45 * max(len(tokens) - 1, 0)
        rarity_bonus = sum(1.0 / max(lowered_prompt.count(token), 1) for token in set(tokens))
        length_penalty = 0.2 * max(len(tokens) - 3, 0)
        return round(unique_count + keyword_overlap * 1.4 + acronym_bonus + multiword_bonus + rarity_bonus - length_penalty, 4)

    def _is_too_close_to_query(self, term: str, lowered_query: str) -> bool:
        """Avoid adding near-duplicates of phrases already in the prompt."""
        term_tokens = {token for token in re.findall(r"\b\w+\b", term.lower()) if token not in STOPWORDS}
        query_tokens = {token for token in re.findall(r"\b\w+\b", lowered_query) if token not in STOPWORDS}
        if not term_tokens:
            return True
        overlap = len(term_tokens & query_tokens) / max(len(term_tokens), 1)
        return overlap >= 0.75

    def _build_blend_phrase(self, related_terms: list[str]) -> str:
        """Create a natural-language clause for blending related terms into prompts."""
        if not related_terms:
            return ""
        trimmed = related_terms[:3]
        if len(trimmed) == 1:
            joined = trimmed[0]
        elif len(trimmed) == 2:
            joined = f"{trimmed[0]} and {trimmed[1]}"
        else:
            joined = f"{trimmed[0]}, {trimmed[1]}, and {trimmed[2]}"
        return f"while covering related concepts such as {joined}"

    def _embedding_model_available(self) -> bool:
        """Report whether sentence-transformers is importable."""
        return SentenceTransformer is not None and util is not None

    def _load_embedding_model(self) -> Any:
        """Lazy-load the sentence-transformers embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
