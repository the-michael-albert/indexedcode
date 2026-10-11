"""Summarize -> embed contract: summaries are the embedding input, a failed
summary never degrades to embedding raw source, class summaries persist, and
missing summaries/vectors are backfilled independently."""

import pytest

from indexedcode import retrieval, semantic
from indexedcode.pipeline import process_next_job, run_until_idle
from indexedcode.reconcile import reconcile
from indexedcode.dal import enqueue_job

from tests.conftest import FakeEmbedder, FakeSummarizer

EMB = "sha256:" + "e" * 64
SUM = "sha256:" + "5" * 64

SRC = (
    "class Box:\n"
    "    def put(self, x):\n"
    "        return x\n"
    "\n"
    "\n"
    "def helper():\n"
    "    return 1\n"
)


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "a.py").write_text(SRC)
    return tmp_path


def index(conn, repo, embed=None, summarize=None, now_ms=1000,
          emb=EMB, summ=SUM):
    reconcile(conn, str(repo), now_ms=now_ms,
              model_fingerprint=emb if embed else None,
              summary_fingerprint=summ if summarize else None)
    return run_until_idle(conn, str(repo), embed=embed, summarize=summarize,
                          now_ms=now_ms, model_fingerprint=emb,
                          summary_fingerprint=summ)


def summaries(conn):
    return {r["qualified_name"]: r["summary"] for table in
            ("functions", "classes") for r in conn.execute(
                f"SELECT qualified_name, summary FROM {table}")}


def test_embedder_receives_generated_summaries_never_source(db, repo):
    embed = FakeEmbedder()
    index(db, repo, embed,
          lambda texts: [f"Plain English summary #{i}"
                         for i, _ in enumerate(texts)])
    embedded = [t for call in embed.calls for t in call]
    assert sorted(embedded) == [f"Plain English summary #{i}"
                                for i in range(3)]
    assert set(summaries(db)) == {"Box", "Box.put", "helper"}
    assert all(summaries(db).values())  # classes included


def test_summarize_runs_before_embed(db, repo):
    order = []
    index(db, repo,
          lambda texts: order.append("embed") or [[1.0, 0.0, 0.0]] * len(texts),
          lambda texts: order.append("summarize") or ["s"] * len(texts))
    assert order == ["summarize", "embed"]


def test_embed_without_summarize_is_refused(db, repo):
    enqueue_job(db, command="refresh_file", folder="", file="a.py",
                now_ms=1)
    with pytest.raises(ValueError, match="requires summarization"):
        process_next_job(db, str(repo), embed=FakeEmbedder(), now_ms=2)


def test_failed_summary_blocks_embedding_and_is_retryable(db, repo):
    embed = FakeEmbedder()
    reconcile(db, str(repo), now_ms=1000)
    outcome = process_next_job(
        db, str(repo), embed=embed,
        summarize=FakeSummarizer(error=RuntimeError("summarizer down")),
        now_ms=1001)
    assert outcome == "retried"
    assert embed.calls == []  # raw source was not embedded as a fallback
    assert db.execute("SELECT COUNT(*) c FROM entity_index").fetchone()["c"] == 0
    assert db.execute("SELECT COUNT(*) c FROM functions").fetchone()["c"] == 0
    job = db.execute("SELECT status, last_error FROM jobs").fetchone()
    assert job["status"] == "retry"
    assert "summarizer down" in job["last_error"]


def test_empty_summary_is_a_failure_not_an_embedding_input(db, repo):
    embed = FakeEmbedder()
    reconcile(db, str(repo), now_ms=1000)
    outcome = process_next_job(db, str(repo), embed=embed,
                               summarize=lambda t: [""] * len(t),
                               now_ms=1001)
    assert outcome == "retried" and embed.calls == []


def test_unchanged_summaries_survive_unrelated_refresh(db, repo):
    summarize = FakeSummarizer()
    index(db, repo, FakeEmbedder(), summarize)
    before = summaries(db)
    (repo / "a.py").write_text(SRC.replace("return 1", "return 2"))
    summarize2, embed2 = FakeSummarizer(), FakeEmbedder()
    index(db, repo, embed2, summarize2, now_ms=5000)
    after = summaries(db)
    assert after["Box"] == before["Box"]
    assert after["Box.put"] == before["Box.put"]
    # only the edited function was re-summarized and re-embedded
    assert len(summarize2.calls) == 1 and len(summarize2.calls[0]) == 1
    assert len(embed2.calls) == 1 and len(embed2.calls[0]) == 1


def test_missing_summaries_backfill_when_vectors_already_exist(db, repo):
    index(db, repo, FakeEmbedder(), FakeSummarizer())
    db.execute("UPDATE functions SET summary = NULL,"
               " summary_input_hash = NULL")
    db.execute("UPDATE classes SET summary = NULL, summary_input_hash = NULL")
    # vectors are intact, but summaries are gone: summary backfill alone
    # must re-queue the unchanged file
    result = reconcile(db, str(repo), now_ms=7000, summary_fingerprint=SUM)
    assert result.refresh_enqueued == 1
    summarize = FakeSummarizer()
    run_until_idle(db, str(repo), summarize=summarize, now_ms=7000,
                   summary_fingerprint=SUM)
    assert all(summaries(db).values())
    assert len(summarize.calls[0]) == 3


def test_raw_source_vectors_are_recomputed_from_summaries(db, repo):
    """Vectors made from source (alpha 3) are stale under this contract."""
    index(db, repo, FakeEmbedder(), FakeSummarizer())
    db.execute("UPDATE entity_index SET input_hash = 'sha256-of-raw-source'")
    result = reconcile(db, str(repo), now_ms=8000, model_fingerprint=EMB,
                       summary_fingerprint=SUM)
    assert result.refresh_enqueued == 1
    embed, summarize = FakeEmbedder(), FakeSummarizer()
    run_until_idle(db, str(repo), embed=embed, summarize=summarize,
                   now_ms=8000, model_fingerprint=EMB,
                   summary_fingerprint=SUM)
    assert summarize.calls == []  # summaries were already current
    assert len(embed.calls[0]) == 3
    assert all(t.startswith("summary of") for t in embed.calls[0])
    done = reconcile(db, str(repo), now_ms=9000, model_fingerprint=EMB,
                     summary_fingerprint=SUM)
    assert done.refresh_enqueued == 0


def test_summary_model_change_resummarizes_and_reembeds(db, repo):
    index(db, repo, FakeEmbedder(), FakeSummarizer())
    other = "sha256:" + "9" * 64
    embed, summarize = FakeEmbedder(), FakeSummarizer()
    index(db, repo, embed, summarize, now_ms=6000, summ=other)
    assert len(summarize.calls[0]) == 3 and len(embed.calls[0]) == 3


def test_status_reports_inference_separately_from_structure(db, repo):
    index(db, repo)  # structural only
    status = retrieval.get_status(db, str(repo), inference_status={
        "summary": {"state": "unavailable", "detail": "mlx_lm missing"},
        "embedding": {"state": "unavailable", "detail": "mlx_embeddings"},
    })
    assert status["coverage"]["ready_files"] == 1
    assert status["pending_jobs"] == 0
    inf = status["inference"]
    assert inf["complete"] is False
    assert inf["stages"]["summary"] == {
        "state": "unavailable", "detail": "mlx_lm missing",
        "done": 0, "total": 3}
    index(db, repo, FakeEmbedder(), FakeSummarizer(), now_ms=3000)
    done = retrieval.get_status(db, str(repo))["inference"]
    assert done["complete"] is True
    assert done["stages"]["embedding"]["done"] == 3


# --- startup readiness -------------------------------------------------------


def test_check_runtime_names_missing_modules_and_install_hint(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="mlx_embeddings") as exc:
        semantic.check_runtime("embedding", {"mode": "managed",
                                             "backend": "mlx"})
    assert "uv tool install" in str(exc.value)
    semantic.check_runtime("embedding", {"mode": "remote", "backend": "mlx"})


def test_start_reports_every_failing_stage(monkeypatch):
    def check(role, cfg):
        raise RuntimeError(f"{role} runtime missing")

    monkeypatch.setattr(semantic, "check_runtime", check)
    with pytest.raises(semantic.InferenceUnavailable) as exc:
        semantic.start_semantic_index()
    assert exc.value.stages == {"summary": "summary runtime missing",
                                "embedding": "embedding runtime missing"}


def test_try_start_reports_stage_status(monkeypatch):
    monkeypatch.setattr(semantic, "check_runtime",
                        lambda role, cfg: (_ for _ in ()).throw(
                            RuntimeError(f"{role} missing")))
    seen = []
    assert semantic.try_start_semantic_index(on_status=seen.append) is None
    assert seen[0]["summary"] == {"state": "unavailable",
                                  "detail": "summary missing"}
    disabled = []
    semantic.try_start_semantic_index(enabled=False,
                                      on_status=disabled.append)
    assert disabled[0]["embedding"]["state"] == "disabled"
