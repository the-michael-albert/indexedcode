"""Semantic indexing wiring: embedding worker lifecycle and callbacks.

Builds the ``embed`` callable shared by ingestion (document vectors) and
search (query vectors) from the configured embedding worker, so both sides
always use the same model and fingerprint.
"""

from __future__ import annotations

import logging
import os

from indexedcode import embeddings
from indexedcode.config import DEFAULTS, load_config
from indexedcode.workers.client import WorkerClient
from indexedcode.workers.factory import build_worker

logger = logging.getLogger(__name__)

# loading weights (and the first download) can take far longer than the
# default 10 s request timeout
EMBED_TIMEOUT_S = 300.0
STARTUP_TIMEOUT_S = 120.0
NORMALIZE_TOLERANCE = 1e-4


class SemanticIndex:
    """A running embedding worker plus the callbacks derived from it."""

    def __init__(self, worker, client, fingerprint, max_batch_items):
        self.worker = worker
        self.client = client
        self.fingerprint = fingerprint
        self.max_batch_items = max_batch_items

    def embed(self, texts):
        """Embed texts in bounded batches; vectors are validated."""
        texts = list(texts)
        vectors = []
        for start in range(0, len(texts), self.max_batch_items):
            batch = texts[start:start + self.max_batch_items]
            response = self.client.embed(batch)
            if len(response.vectors) != len(batch):
                raise embeddings.EmbeddingValidationError(
                    f"worker returned {len(response.vectors)} vectors for"
                    f" {len(batch)} texts")
            for vector in response.vectors:
                embeddings.validate(vector, response.dimension,
                                    tolerance=NORMALIZE_TOLERANCE)
            vectors.extend(response.vectors)
        return vectors

    def poll(self):
        poll = getattr(self.worker, "poll", None)
        if poll is not None:
            poll()

    def stop(self):
        stop = getattr(self.worker, "stop", None)
        if stop is not None:
            stop()


def start_semantic_index(config_path=None) -> SemanticIndex:
    """Start (or connect to) the embedding worker described by the config.

    Raises on any failure; callers decide whether to continue without
    semantic search.
    """
    if config_path:
        cfg = load_config(config_path)["embedding"]
    else:
        cfg = dict(DEFAULTS["embedding"])
    worker = build_worker("embedding", cfg)
    if hasattr(worker, "start"):
        worker.startup_timeout_s = STARTUP_TIMEOUT_S
        worker.start()
        client = WorkerClient(
            base_url=worker.base_url, secret=worker.secret,
            expected_fingerprint=worker.expected_fingerprint,
            timeout_s=EMBED_TIMEOUT_S,
        )
        fingerprint = worker.model_info.fingerprint
    else:
        client = worker.client
        client.timeout_s = EMBED_TIMEOUT_S
        fingerprint = client.model_info().fingerprint
    semantic = SemanticIndex(worker, client, fingerprint,
                             cfg.get("max_batch_items", 32))
    try:
        # prove the runtime works (and load weights once) before indexing
        semantic.embed(["warmup"])
    except Exception:
        semantic.stop()
        raise
    return semantic


def try_start_semantic_index(config_path=None, enabled=True):
    """start_semantic_index() that degrades loudly instead of raising."""
    if not enabled:
        logger.info("semantic indexing disabled (--no-embeddings);"
                    " search is symbol/lexical only")
        return None
    try:
        semantic = start_semantic_index(config_path)
    except Exception as exc:  # noqa: BLE001 - any startup failure degrades
        logger.warning(
            "semantic indexing unavailable (%s); continuing with"
            " symbol/lexical search only. Install the runtime with"
            " `pip install 'indexedcode[mlx]'` (Apple Silicon) or"
            " `pip install 'indexedcode[cpu]'`, or pass --no-embeddings.",
            exc)
        return None
    logger.info("semantic indexing enabled (model fingerprint %s)",
                semantic.fingerprint)
    return semantic
