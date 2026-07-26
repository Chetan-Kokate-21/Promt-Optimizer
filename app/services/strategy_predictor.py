"""ML-based strategy prediction for prompt optimization tactics.

This module exposes a class-based Random Forest predictor that learns from a
synthetic dataset and recommends the most suitable optimization strategies for
an incoming prompt. It is designed to remain self-contained so the project can
evolve from mock data to real training data without changing the service API.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from app.services.strategy_feature_extractor import (
    DOMAIN_CODING,
    DOMAIN_GENERAL,
    DOMAIN_MATH,
    INTENT_ANALYZE,
    INTENT_CLASSIFY,
    INTENT_EXPLAIN,
    INTENT_GENERATE,
    INTENT_OPTIMIZE,
    INTENT_OTHER,
    INTENT_SOLVE,
    INTENT_SUMMARIZE,
    PromptFeatures,
    StrategyFeatureExtractor,
)


COST_STRATEGIES = [
    "direct_instruction",
    "compression",
    "remove_examples",
    "minimal_output",
]

CONTEXT_STRATEGIES = [
    "role_based",
    "few_shot",
    "chain_of_thought",
    "constraint_based",
]

ALL_STRATEGIES = COST_STRATEGIES + CONTEXT_STRATEGIES


@dataclass(slots=True)
class StrategyPrediction:
    """Structured prediction output for downstream pipeline consumers."""

    top_strategies: list[str]
    ranked_strategies: list[dict[str, float]]
    features: dict[str, Any]


class RandomForestStrategyPredictor:
    """Train and serve multi-label strategy recommendations."""

    def __init__(self, random_state: int = 42, dataset_size: int = 320) -> None:
        """Initialize the predictor and fit it on synthetic training data."""
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=2,
                random_state=random_state,
            )
        )
        self._fit()

    def predict(self, features: PromptFeatures, top_k: int = 3) -> StrategyPrediction:
        """Predict the top optimization strategies for extracted prompt features."""
        feature_vector = [features.to_model_vector()]
        probability_map = self._predict_probabilities(feature_vector)[0]
        ranked = sorted(probability_map.items(), key=lambda item: item[1], reverse=True)

        selected = [name for name, score in ranked if score >= 0.45][:top_k]
        if len(selected) < 2:
            selected = [name for name, _ in ranked[: min(2, len(ranked))]]

        return StrategyPrediction(
            top_strategies=selected,
            ranked_strategies=[
                {"strategy": name, "score": round(score, 4)}
                for name, score in ranked[:top_k]
            ],
            features=features.to_dict(),
        )

    def _fit(self) -> None:
        """Train the Random Forest model on a synthetic feature-label dataset."""
        feature_rows, label_rows = self._build_synthetic_dataset()
        self.model.fit(feature_rows, label_rows)

    def _build_synthetic_dataset(self) -> tuple[list[list[float]], list[list[int]]]:
        """Generate synthetic samples that encode prompt-to-strategy heuristics."""
        rng = random.Random(self.random_state)
        feature_rows: list[list[float]] = []
        label_rows: list[list[int]] = []

        domains = [DOMAIN_CODING, DOMAIN_MATH, DOMAIN_GENERAL]
        intents = [
            INTENT_GENERATE,
            INTENT_EXPLAIN,
            INTENT_SUMMARIZE,
            INTENT_SOLVE,
            INTENT_ANALYZE,
            INTENT_OPTIMIZE,
            INTENT_CLASSIFY,
            INTENT_OTHER,
        ]

        for _ in range(self.dataset_size):
            token_length = rng.randint(8, 450)
            complexity_score = round(rng.uniform(0.2, 9.8), 3)
            ambiguity_score = round(rng.uniform(0.0, 9.2), 3)
            domain = rng.choices(domains, weights=[0.35, 0.2, 0.45], k=1)[0]
            intent_type = rng.choices(
                intents,
                weights=[0.2, 0.16, 0.1, 0.14, 0.14, 0.1, 0.06, 0.1],
                k=1,
            )[0]
            requires_reasoning = bool(
                complexity_score >= 4.7
                or domain in {DOMAIN_CODING, DOMAIN_MATH}
                or intent_type in {INTENT_ANALYZE, INTENT_SOLVE, INTENT_OPTIMIZE}
            )
            features = PromptFeatures(
                token_length=token_length,
                complexity_score=complexity_score,
                ambiguity_score=ambiguity_score,
                domain=domain,
                intent_type=intent_type,
                requires_reasoning=requires_reasoning,
            )

            labels = self._label_sample(features)
            feature_rows.append(features.to_model_vector())
            label_rows.append(labels)

        return feature_rows, label_rows

    def _label_sample(self, features: PromptFeatures) -> list[int]:
        """Generate synthetic labels using deterministic strategy heuristics."""
        selected: set[str] = set()

        if features.token_length <= 45 and features.ambiguity_score <= 2.5:
            selected.add("direct_instruction")

        if features.token_length >= 120 or features.intent_type == INTENT_SUMMARIZE:
            selected.add("compression")

        if features.token_length >= 180 or (
            features.ambiguity_score <= 2.0 and features.intent_type in {INTENT_SOLVE, INTENT_EXPLAIN}
        ):
            selected.add("remove_examples")

        if features.intent_type in {INTENT_CLASSIFY, INTENT_SUMMARIZE} or features.token_length <= 25:
            selected.add("minimal_output")

        if features.domain in {DOMAIN_CODING, DOMAIN_GENERAL} and features.intent_type in {
            INTENT_GENERATE,
            INTENT_EXPLAIN,
            INTENT_ANALYZE,
        }:
            selected.add("role_based")

        if features.domain == DOMAIN_CODING or (
            features.domain == DOMAIN_MATH and features.intent_type in {INTENT_SOLVE, INTENT_EXPLAIN}
        ):
            selected.add("few_shot")

        if features.requires_reasoning and features.complexity_score >= 4.0:
            selected.add("chain_of_thought")

        if features.ambiguity_score >= 3.2 or features.intent_type == INTENT_OPTIMIZE:
            selected.add("constraint_based")

        if not selected:
            selected.add("direct_instruction")
            if features.requires_reasoning:
                selected.add("chain_of_thought")

        return [1 if strategy in selected else 0 for strategy in ALL_STRATEGIES]

    def _predict_probabilities(self, feature_rows: list[list[float]]) -> list[dict[str, float]]:
        """Return per-strategy positive-class probabilities for each sample."""
        estimator_probabilities = self.model.predict_proba(feature_rows)
        predictions: list[dict[str, float]] = []

        for row_index in range(len(feature_rows)):
            score_map: dict[str, float] = {}
            for strategy, probabilities in zip(ALL_STRATEGIES, estimator_probabilities):
                if len(probabilities[row_index]) == 1:
                    positive_score = float(probabilities[row_index][0])
                else:
                    positive_score = float(probabilities[row_index][1])
                score_map[strategy] = positive_score
            predictions.append(score_map)

        return predictions


class StrategyPredictorService:
    """Service wrapper that extracts features and predicts top strategies."""

    def __init__(
        self,
        feature_extractor: StrategyFeatureExtractor | None = None,
        predictor: RandomForestStrategyPredictor | None = None,
    ) -> None:
        """Initialize the strategy predictor service dependencies."""
        self.feature_extractor = feature_extractor or StrategyFeatureExtractor()
        self.predictor = predictor or RandomForestStrategyPredictor()

    def predict(self, prompt_data: dict[str, Any]) -> dict[str, Any]:
        """Predict the best prompt optimization strategies for a request."""
        features = self.feature_extractor.extract(prompt_data)
        prediction = self.predictor.predict(features)
        return {
            "features": prediction.features,
            "recommended_strategies": prediction.top_strategies,
            "ranked_strategies": prediction.ranked_strategies,
            "strategy_space": {
                "cost_strategies": COST_STRATEGIES,
                "context_strategies": CONTEXT_STRATEGIES,
            },
        }
