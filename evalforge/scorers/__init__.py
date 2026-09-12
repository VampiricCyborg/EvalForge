"""Scorer implementations for EvalForge."""

from evalforge.scorers.base import Scorer
from evalforge.scorers.exact_match import ExactMatchScorer
from evalforge.scorers.llm_judge import LLMJudgeScorer
from evalforge.scorers.semantic import SemanticSimilarityScorer

SCORERS: dict[str, type[Scorer]] = {
    ExactMatchScorer.name: ExactMatchScorer,
    SemanticSimilarityScorer.name: SemanticSimilarityScorer,
    LLMJudgeScorer.name: LLMJudgeScorer,
}

__all__ = [
    "Scorer",
    "ExactMatchScorer",
    "SemanticSimilarityScorer",
    "LLMJudgeScorer",
    "SCORERS",
]
