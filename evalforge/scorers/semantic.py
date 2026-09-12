"""Semantic similarity scoring via local sentence-transformer embeddings."""

import time

from sentence_transformers import SentenceTransformer, util

from evalforge.schema import EvalResult, ModelOutput, TestCase
from evalforge.scorers.base import Scorer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSimilarityScorer(Scorer):
    """Scores cosine similarity between reference and output embeddings.

    Uses a small local sentence-transformers model (no API calls, no cost).
    Cosine similarity is clamped to [0, 1] for the primary score; the raw,
    unclamped value is kept in `raw_output` for diagnostics.
    """

    name = "semantic"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def score(self, test_case: TestCase, output: ModelOutput) -> EvalResult:
        if test_case.reference is None:
            raise ValueError(
                f"test_case '{test_case.id}' has no reference; "
                f"{self.name} requires one to score against"
            )

        start = time.perf_counter()
        embeddings = self._model.encode(
            [test_case.reference, output.output], convert_to_tensor=True
        )
        raw_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        clamped_score = max(0.0, min(1.0, raw_similarity))
        latency_ms = (time.perf_counter() - start) * 1000

        return EvalResult(
            test_case_id=test_case.id,
            method=self.name,
            score=clamped_score,
            latency_ms=latency_ms,
            raw_output={
                "raw_cosine_similarity": raw_similarity,
                "model": self._model_name,
            },
        )
