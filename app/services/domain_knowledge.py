"""Shared domain guidance used by prompt optimization stages."""

from __future__ import annotations

import re
from typing import Any


_DOMAIN_PROFILES: dict[str, dict[str, Any]] = {
    "data": {
        "keywords": ["data analytics", "etl", "pipeline", "warehouse", "sandbox", "lineage", "quality"],
        "guidance": "Use data-platform terminology, mention pipelines, data quality, lineage, staging, and analytics workflow details.",
        "constraint": "Preserve data-engineering accuracy, mention pipeline stages and data-quality considerations, and prefer concrete platform terminology.",
        "example": "Input: ETL issue in analytics sandbox | Output: pipeline stages, validation checks, and warehouse-oriented debugging guidance",
        "role": "Act as a senior data engineer and analytics specialist.",
    },
    "coding": {
        "keywords": ["code", "api", "function", "debugging", "edge cases", "complexity", "implementation"],
        "guidance": "Use precise software-engineering language, mention implementation details, failure modes, and edge cases.",
        "constraint": "Preserve technical correctness, mention relevant APIs or language constructs, and prefer concrete implementation detail.",
        "example": "Input: vague coding issue | Output: debugging steps, likely root causes, and code-level guidance",
        "role": "Act as a senior software engineer.",
    },
    "math": {
        "keywords": ["equation", "derivation", "proof", "variables", "formula", "steps", "verification"],
        "guidance": "Use mathematical vocabulary, define variables, and emphasize derivation, verification, and step-by-step reasoning.",
        "constraint": "Preserve mathematical correctness, show intermediate reasoning, and verify the final result where possible.",
        "example": "Input: algebra problem | Output: formulas, worked steps, and final verification",
        "role": "Act as a careful math tutor.",
    },
    "finance": {
        "keywords": ["cash flow", "revenue", "costs", "risk", "forecast", "ROI", "assumptions"],
        "guidance": "Use finance-specific terminology, highlight assumptions, tradeoffs, and quantitative business impact.",
        "constraint": "Preserve financial accuracy, state assumptions clearly, and discuss risks or tradeoffs when relevant.",
        "example": "Input: budgeting request | Output: assumptions, cost drivers, and decision-ready financial summary",
        "role": "Act as a finance analyst.",
    },
    "healthcare": {
        "keywords": ["symptoms", "diagnosis", "treatment", "risk factors", "history", "evidence", "safety"],
        "guidance": "Use healthcare terminology carefully, organize clinical context, and emphasize evidence and safety.",
        "constraint": "Preserve safety-sensitive details, avoid unsupported claims, and keep the explanation clinically grounded.",
        "example": "Input: symptom question | Output: organized clinical explanation, red flags, and next-step guidance",
        "role": "Act as a careful healthcare educator.",
    },
    "legal": {
        "keywords": ["statute", "clause", "liability", "jurisdiction", "obligation", "risk", "compliance"],
        "guidance": "Use legal terminology, identify obligations and risks, and structure the answer with careful scope and qualifiers.",
        "constraint": "Preserve legal nuance, mention scope limits, and avoid overstating certainty.",
        "example": "Input: contract clause question | Output: issue summary, obligations, risks, and careful interpretation",
        "role": "Act as a precise legal analyst.",
    },
    "writing": {
        "keywords": ["audience", "voice", "structure", "clarity", "draft", "narrative", "revision"],
        "guidance": "Use writing-focused language, tailor for audience and voice, and improve structure and flow.",
        "constraint": "Preserve intent, audience, tone, and narrative clarity.",
        "example": "Input: rough paragraph | Output: revised draft with clearer structure and audience-appropriate tone",
        "role": "Act as an expert writing coach.",
    },
    "science": {
        "keywords": ["hypothesis", "evidence", "mechanism", "variables", "method", "results", "limitations"],
        "guidance": "Use scientific vocabulary, organize claims around evidence and mechanisms, and mention limitations when useful.",
        "constraint": "Preserve scientific accuracy, distinguish evidence from inference, and mention limitations where relevant.",
        "example": "Input: science concept | Output: mechanism, evidence, and limits of the explanation",
        "role": "Act as a careful scientific explainer.",
    },
    "general": {
        "keywords": ["context", "constraints", "examples", "details", "clarity"],
        "guidance": "Use the user's domain terms where possible and add structure that improves clarity and relevance.",
        "constraint": "Preserve the user's intent and keep the response grounded in the task.",
        "example": "Input: vague request | Output: clearer task, useful context, and structured response guidance",
        "role": "Act as a helpful domain expert.",
    },
}

_DOMAIN_HINTS: list[tuple[str, set[str]]] = [
    ("data", {"data", "analytics", "etl", "warehouse", "sandbox", "pipeline", "lineage", "airflow", "dbt", "spark"}),
    ("coding", {"python", "code", "api", "function", "debug", "bug", "sql", "javascript", "java", "flask"}),
    ("math", {"math", "equation", "algebra", "geometry", "calculus", "integral", "derivative", "proof"}),
    ("finance", {"finance", "budget", "forecast", "revenue", "profit", "cashflow", "roi", "investment"}),
    ("healthcare", {"health", "healthcare", "medical", "symptom", "diagnosis", "treatment", "patient", "clinical"}),
    ("legal", {"legal", "law", "contract", "clause", "liability", "compliance", "statute", "jurisdiction"}),
    ("writing", {"write", "writing", "essay", "article", "story", "tone", "copy", "draft"}),
    ("science", {"science", "biology", "chemistry", "physics", "experiment", "hypothesis", "mechanism"}),
]


def infer_domain(prompt: str, metadata: dict[str, Any] | None = None) -> str:
    """Infer a richer domain label from text and optional metadata."""
    if metadata:
        explicit = str(metadata.get("domain", "")).strip().lower()
        if explicit in _DOMAIN_PROFILES:
            return explicit

    lowered = prompt.lower()
    tokens = set(re.findall(r"\b\w+\b", lowered))
    for domain, hints in _DOMAIN_HINTS:
        if tokens & hints:
            return domain
    return "general"


def get_domain_profile(domain: str) -> dict[str, Any]:
    """Return a normalized domain profile with shared guidance."""
    normalized = str(domain or "general").strip().lower()
    return dict(_DOMAIN_PROFILES.get(normalized, _DOMAIN_PROFILES["general"]))
