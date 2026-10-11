"""Semantic indexing wiring: embedding worker lifecycle and callbacks.

Builds the ``embed`` callable shared by ingestion (document vectors) and
search (query vectors) from the configured embedding worker, so both sides
always use the same model and fingerprint.
"""

from __future__ import annotations

import importlib.util
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


SUMMARY_TIMEOUT_S = 600.0

# importable modules each managed backend needs, per inference role
_RUNTIME_MODULES = {
    ("mlx", "embedding"): ("mlx_embeddings",),
    ("mlx", "summary"): ("mlx_lm",),
    ("rocm", "embedding"): ("torch", "transformers"),
    ("rocm", "summary"): ("torch", "transformers"),
    ("cpu", "embedding"): ("torch", "transformers"),
    ("cpu", "summary"): ("torch", "transformers"),
}
_INSTALL_HINT = {
    "mlx": "`uv tool install --force --with mlx-embeddings --with mlx-lm"
           " indexedcode` (or `pip install 'indexedcode[mlx]'`); MLX needs"
           " Apple Silicon",
    "rocm": "`pip install torch transformers` for your ROCm build",
    "cpu": "`uv tool install --with torch --with transformers indexedcode`"
           " (or `pip install 'indexedcode[cpu]'`)",
}

SUMMARY_PROMPT = (
    "Summarize what the following code does in one or two plain-English"
    " sentences for a developer searching a codebase. Mention its purpose,"
    " inputs and outputs. Reply with the summary only.\n\n```\n{code}\n```\n"
    "Summary:")


class InferenceUnavailable(RuntimeError):
    """One or more required inference stages could not start.

    ``stages`` maps each stage (``summary``, ``embedding``) to ``None``
    (started) or the reason it could not start.
    """

    def __init__(self, stages):
        self.stages = dict(stages)
        super().__init__("; ".join(
            f"{name}: {reason}" for name, reason in self.stages.items()
            if reason))


def check_runtime(role, section_cfg):
    """Raise if a managed worker's runtime is missing from *this*
    interpreter (workers run under ``sys.executable``)."""
    if section_cfg["mode"] != "managed":
        return
    backend = section_cfg["backend"]
    missing = [m for m in _RUNTIME_MODULES.get((backend, role), ())
               if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            f"the {backend} {role} runtime is not installed in this"
            f" environment (missing: {', '.join(missing)}); install with"
            f" {_INSTALL_HINT[backend]}")


class SemanticIndex:
    """Running summary and embedding workers plus their callbacks.

    Indexing is summarize -> embed: entity source is summarized and the
    generated summary is what gets embedded. Queries are embedded directly.
    """

    def __init__(self, worker, client, fingerprint, max_batch_items,
                 summary_worker=None, summary_client=None,
                 summary_fingerprint=None, max_output_tokens=256):
        self.worker = worker
        self.client = client
        self.fingerprint = fingerprint
        self.max_batch_items = max_batch_items
        self.summary_worker = summary_worker
        self.summary_client = summary_client
        self.summary_fingerprint = summary_fingerprint
        self.max_output_tokens = max_output_tokens

    def summarize(self, texts):
        """One summary per source text, in order; empty summaries raise."""
        summaries = []
        for text in texts:
            response = self.summary_client.summarize(
                SUMMARY_PROMPT.format(code=text), self.max_output_tokens)
            summary = (response.summary or "").strip()
            if not summary:
                raise RuntimeError("summary worker returned an empty summary")
            summaries.append(summary)
        return summaries

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

    def _workers(self):
        return [w for w in (self.worker, self.summary_worker)
                if w is not None]

    def poll(self):
        for worker in self._workers():
            poll = getattr(worker, "poll", None)
            if poll is not None:
                poll()

    def stop(self):
        for worker in self._workers():
            stop = getattr(worker, "stop", None)
            if stop is not None:
                stop()


def _start_stage(name, role, cfg, timeout_s):
    """Start/connect one worker; returns (worker, client, fingerprint)."""
    check_runtime(role, cfg)
    worker = build_worker(name, cfg)
    try:
        if hasattr(worker, "start"):
            worker.startup_timeout_s = STARTUP_TIMEOUT_S
            worker.start()
            client = WorkerClient(
                base_url=worker.base_url, secret=worker.secret,
                expected_fingerprint=worker.expected_fingerprint,
                timeout_s=timeout_s,
            )
            fingerprint = worker.model_info.fingerprint
        else:
            client = worker.client
            client.timeout_s = timeout_s
            fingerprint = client.model_info().fingerprint
    except Exception:
        _stop_quietly(worker)
        raise
    return worker, client, fingerprint


def _stop_quietly(worker):
    stop = getattr(worker, "stop", None)
    if stop is not None:
        try:
            stop()
        except Exception:  # noqa: BLE001 - best effort during teardown
            pass


def start_semantic_index(config_path=None) -> SemanticIndex:
    """Start (or connect to) the summary and embedding workers.

    Both stages are required: every stage is attempted so the failure
    reports every reason, then :class:`InferenceUnavailable` is raised if
    either could not start. Callers decide whether to continue without
    semantic search.
    """
    if config_path:
        config = load_config(config_path)
    else:
        config = {k: dict(v) for k, v in DEFAULTS.items()}
    emb_cfg, sum_cfg = config["embedding"], config["summarization"]

    stages, started = {}, {}
    for stage, name, role, cfg, timeout in (
            ("summary", "summarization", "summary", sum_cfg,
             SUMMARY_TIMEOUT_S),
            ("embedding", "embedding", "embedding", emb_cfg,
             EMBED_TIMEOUT_S)):
        try:
            started[stage] = _start_stage(name, role, cfg, timeout)
            stages[stage] = None
        except Exception as exc:  # noqa: BLE001 - reported per stage
            stages[stage] = str(exc) or type(exc).__name__

    def abort(exc):
        for worker, _client, _fp in started.values():
            _stop_quietly(worker)
        raise exc

    if any(stages.values()):
        abort(InferenceUnavailable(stages))

    sum_worker, sum_client, sum_fp = started["summary"]
    worker, client, fingerprint = started["embedding"]
    semantic = SemanticIndex(
        worker, client, fingerprint, emb_cfg.get("max_batch_items", 32),
        summary_worker=sum_worker, summary_client=sum_client,
        summary_fingerprint=sum_fp,
        max_output_tokens=sum_cfg.get("max_output_tokens", 256))
    try:
        # prove both runtimes work (and load weights once) before indexing
        try:
            semantic.summarize(["def warmup(): pass"])
        except Exception as exc:  # noqa: BLE001
            stages["summary"] = f"warmup failed: {exc}"
        try:
            semantic.embed(["warmup"])
        except Exception as exc:  # noqa: BLE001
            stages["embedding"] = f"warmup failed: {exc}"
        if any(stages.values()):
            raise InferenceUnavailable(stages)
    except Exception:
        semantic.stop()
        raise
    return semantic


def try_start_semantic_index(config_path=None, enabled=True,
                             on_status=None):
    """start_semantic_index() that degrades loudly instead of raising.

    ``on_status`` receives ``{"summary": state, "embedding": state}`` where
    state is ``{"state": "ready"|"unavailable"|"disabled", "detail": ...}``
    so callers can surface per-stage readiness.
    """
    def report(state, detail=None, stages=None):
        if on_status is None:
            return
        stages = stages or {"summary": None, "embedding": None}
        on_status({
            stage: ({"state": state, "detail": detail} if state == "disabled"
                    else {"state": "unavailable", "detail": reason}
                    if reason else {"state": state, "detail": None})
            for stage, reason in stages.items()
        })

    if not enabled:
        logger.info("semantic indexing disabled (--no-embeddings);"
                    " search is symbol/lexical only")
        report("disabled", "disabled with --no-embeddings")
        return None
    try:
        semantic = start_semantic_index(config_path)
    except InferenceUnavailable as exc:
        for stage, reason in exc.stages.items():
            logger.warning("%s stage unavailable: %s", stage,
                           reason or "not started because the other stage"
                           " failed")
        logger.warning(
            "semantic indexing unavailable; continuing with symbol/lexical"
            " search only (summarize -> embed needs both stages).")
        report("unavailable", stages=exc.stages)
        return None
    except Exception as exc:  # noqa: BLE001 - any startup failure degrades
        logger.warning(
            "semantic indexing unavailable (%s); continuing with"
            " symbol/lexical search only. Pass --no-embeddings to silence"
            " this.", exc)
        reason = str(exc) or type(exc).__name__
        report("unavailable", stages={"summary": reason,
                                      "embedding": reason})
        return None
    logger.info("semantic indexing enabled (summary model %s, embedding"
                " model %s)", semantic.summary_fingerprint,
                semantic.fingerprint)
    report("ready")
    return semantic
