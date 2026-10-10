"""Operations documentation (plan Phase 6)."""

from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "docs" / "ops"


def test_local_mlx_example_exists():
    text = (OPS / "local-mlx.md").read_text()
    assert "mlx" in text.lower()
    assert "orchestrator serve" in text


def test_remote_rocm_example_exists():
    text = (OPS / "remote-rocm.md").read_text()
    assert "rocm" in text.lower()
    assert "remote" in text.lower()
    assert "https" in text  # authenticated encrypted transport


def test_service_manager_units_exist():
    text = (OPS / "service-units.md").read_text()
    assert "launchd" in text or "systemd" in text


def test_backup_rebuild_rollback_procedure_exists():
    text = (OPS / "backup-rebuild-rollback.md").read_text()
    for keyword in ("backup", "rebuild", "rollback", "wal"):
        assert keyword in text.lower()


def test_memory_profile_expectations_documented():
    text = (OPS / "local-mlx.md").read_text()
    assert "1-2.5" in text or "1–2.5" in text  # measured target profile
