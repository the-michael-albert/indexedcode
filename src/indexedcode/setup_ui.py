"""Interactive Textual setup for starting a repository watcher."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Switch,
)

from indexedcode.config import ConfigError, load_config
from indexedcode.workers.api import PROVIDERS

CONFIG_NAME = "indexedcode.toml"
STAGE_MODES = (("Local models", "local"),
               ("Remote worker pool (Docker)", "remote"),
               ("Hosted API", "api"))
STAGES = (("summarization", "Summarization", "summary"),
          ("embedding", "Embedding", "embedding"))


@dataclass
class StageSettings:
    """Connection settings for one inference stage."""

    mode: str = "local"          # local | remote | api
    provider: str = "openai"     # api only
    url: str = ""                # remote worker URL, or API base URL
    api_key: str = ""            # remote bearer token, or provider API key
    model: str = ""              # blank = the mode's default
    parallel: str = ""           # blank = the mode's default


@dataclass
class SemanticSettings:
    """What the setup screen decided about summaries and embeddings."""

    enabled: bool = True
    config_path: str | None = None
    stages: dict = field(default_factory=lambda: {
        name: StageSettings() for name, _, _ in STAGES})


def stage_section(name: str, stage: StageSettings) -> dict | None:
    """The ``indexedcode.toml`` section for a stage, or ``None`` for local."""
    if stage.mode == "local":
        return None
    role = "summary" if name == "summarization" else "embedding"
    section: dict = {}
    if stage.mode == "remote":
        from indexedcode.workers.adapters import default_model

        section.update(mode="remote", backend="vulkan",
                       remote_url=stage.url.strip())
        section["model"] = stage.model.strip() or default_model(
            role, "vulkan")
        if stage.api_key:
            section["remote_token"] = stage.api_key
    else:
        section.update(mode="api", provider=stage.provider)
        if stage.url.strip():
            section["base_url"] = stage.url.strip()
        if stage.model.strip():
            section["model"] = stage.model.strip()
        if stage.api_key:
            section["api_key"] = stage.api_key
    if stage.parallel.strip():
        section["max_concurrent_requests"] = int(stage.parallel)
    return section


def _toml_value(value) -> str:
    return json.dumps(value)  # JSON strings/ints are valid TOML


def render_config(settings: SemanticSettings) -> str | None:
    """TOML text for the non-local stages, or ``None`` if all are local."""
    lines = []
    for name, _label, _role in STAGES:
        section = stage_section(name, settings.stages[name])
        if section is None:
            continue
        lines.append(f"[{name}]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in section.items()]
        lines.append("")
    return "\n".join(lines) if lines else None


def validate_semantic(settings: SemanticSettings, tmp_dir: Path) -> None:
    """Raise ``ValueError`` if the settings would not load; checks the same
    rules as the real loader by round-tripping a temporary file."""
    for name, label, _role in STAGES:
        stage = settings.stages[name]
        if stage.mode == "remote" and not stage.url.strip():
            raise ValueError(f"{label}: enter the worker URL")
        if stage.parallel.strip() and not stage.parallel.strip().isdigit():
            raise ValueError(f"{label}: parallel requests must be a number")
    text = render_config(settings)
    if text is None:
        return
    probe = tmp_dir / f".{CONFIG_NAME}.check"
    probe.write_text(text)
    try:
        load_config(probe)
    except ConfigError as exc:
        raise ValueError(str(exc)) from exc
    finally:
        probe.unlink(missing_ok=True)


def write_config(path: Path, settings: SemanticSettings) -> str | None:
    """Write the config (owner-only: it may hold API keys); returns its
    path, or ``None`` (removing a stale file) when every stage is local."""
    text = render_config(settings)
    if text is None:
        path.unlink(missing_ok=True)
        return None
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o600)
    return str(path)


def load_semantic_settings(path: Path) -> SemanticSettings:
    """Prefill from a config previously written next to the database."""
    settings = SemanticSettings()
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return settings
    for name, _label, _role in STAGES:
        given = raw.get(name, {})
        mode = {"remote": "remote", "api": "api"}.get(given.get("mode"))
        if mode is None:
            continue
        settings.stages[name] = StageSettings(
            mode=mode,
            provider=given.get("provider") or "openai",
            url=given.get("remote_url" if mode == "remote" else "base_url")
            or "",
            api_key=given.get(
                "remote_token" if mode == "remote" else "api_key") or "",
            model=given.get("model") or "",
            parallel=str(given.get("max_concurrent_requests") or ""),
        )
    return settings


def default_db_path(root: str | os.PathLike[str]) -> Path:
    """Return a stable user-owned database path outside the repository."""
    resolved = Path(root).expanduser().resolve()
    data_home = Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    )
    digest = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:8]
    name = "".join(c if c.isalnum() or c in "-_" else "-" for c in resolved.name)
    return data_home / "indexedcode" / f"{name or 'repository'}-{digest}" / "index.sqlite"


def validate_setup(root: str, db: str, port: str) -> tuple[Path, Path, int]:
    """Validate and normalize values without changing the filesystem."""
    repo = Path(root).expanduser().resolve()
    database = Path(db).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError("Repository must be an existing directory")
    if database == repo or repo in database.parents:
        raise ValueError("Store the database outside the watched repository")
    try:
        http_port = int(port)
    except ValueError as exc:
        raise ValueError("HTTP port must be a number") from exc
    if not 0 <= http_port <= 65535:
        raise ValueError("HTTP port must be between 0 and 65535")
    return repo, database, http_port


class SetupApp(App[tuple[str, str, int, SemanticSettings] | None]):
    """Collect a repository, database location, HTTP port, and the
    summarization/embedding connection settings."""

    TITLE = "IndexedCode setup"
    SUB_TITLE = "Choose a repository to index and watch"
    CSS = """
    Screen { align: center middle; }
    #dialog { width: 92%; height: 92%; padding: 1 2; border: round $accent; }
    #form { height: 1fr; }
    #tree { height: 10; margin-bottom: 1; }
    Label { margin-top: 1; }
    Input { width: 1fr; }
    #actions { height: auto; margin-top: 1; align-horizontal: right; }
    #error { color: $error; height: auto; }
    .stage { height: auto; border: round $panel; padding: 0 1; margin-top: 1; }
    .stage-fields { height: auto; }
    .hidden { display: none; }
    Button { margin-left: 1; }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, root: str = ".", db: str | None = None, port: int = 8765):
        super().__init__()
        resolved = Path(root).expanduser().resolve()
        self.initial_root = resolved
        self.initial_db = Path(db).expanduser() if db else default_db_path(resolved)
        self.initial_port = port
        self.initial_semantic = load_semantic_settings(
            self.initial_db.parent / CONFIG_NAME)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dialog"):
            with VerticalScroll(id="form"):
                yield Static(
                    "Select a folder below or edit the repository path. "
                    "The index database stays outside the watched repository."
                )
                yield DirectoryTree(str(Path.home()), id="tree")
                yield Label("Repository")
                yield Input(str(self.initial_root), id="root")
                yield Label("Index database")
                yield Input(str(self.initial_db), id="db")
                yield Label("HTTP port (use 0 to choose a free port)")
                yield Input(str(self.initial_port), id="port", type="integer")
                yield Label("Semantic search (summaries + embeddings)")
                yield Switch(value=True, id="semantic")
                yield Static(
                    "Local models need no setup. A remote worker pool is the"
                    " Docker Compose stack in deploy/docker; a hosted API"
                    " sends code to the provider. Settings are saved next to"
                    " the database (keys: owner-only file).")
                for name, label, _role in STAGES:
                    stage = self.initial_semantic.stages[name]
                    with Vertical(classes="stage", id=f"{name}-box"):
                        yield Label(f"{label}")
                        yield Select(STAGE_MODES, value=stage.mode,
                                     allow_blank=False, id=f"{name}-mode")
                        with Vertical(classes="stage-fields",
                                      id=f"{name}-fields"):
                            with Vertical(classes="stage-fields",
                                          id=f"{name}-provider-box"):
                                yield Label("Provider (hosted API)")
                                yield Select(
                                    [(p, p) for p in PROVIDERS],
                                    value=stage.provider, allow_blank=False,
                                    id=f"{name}-provider")
                            yield Label("URL (worker URL, or API base URL;"
                                        " blank = provider default)")
                            yield Input(stage.url, id=f"{name}-url",
                                        placeholder="http://127.0.0.1:8802")
                            yield Label("API key / bearer token")
                            yield Input(stage.api_key, password=True,
                                        id=f"{name}-key")
                            yield Label("Model (blank = default)")
                            yield Input(stage.model, id=f"{name}-model")
                            yield Label("Parallel requests (blank = default)")
                            yield Input(stage.parallel, id=f"{name}-parallel",
                                        type="integer")
            yield Static("", id="error")
            with Horizontal(id="actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Start watching", id="start", variant="primary")
        yield Footer()

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        root = self.query_one("#root", Input)
        db = self.query_one("#db", Input)
        old_default = default_db_path(root.value)
        root.value = str(event.path)
        if Path(db.value).expanduser() == old_default:
            db.value = str(default_db_path(event.path))

    def on_mount(self) -> None:
        for name, _label, _role in STAGES:
            self._sync_stage(name)

    def on_select_changed(self, event: Select.Changed) -> None:
        for name, _label, _role in STAGES:
            if event.select.id == f"{name}-mode":
                self._sync_stage(name)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "semantic":
            for name, _label, _role in STAGES:
                self._sync_stage(name)

    def _sync_stage(self, name: str) -> None:
        """Show only the fields the chosen mode uses."""
        mode = self.query_one(f"#{name}-mode", Select).value
        on = self.query_one("#semantic", Switch).value
        self.query_one(f"#{name}-box").set_class(not on, "hidden")
        self.query_one(f"#{name}-fields").set_class(mode == "local", "hidden")
        self.query_one(f"#{name}-provider-box").set_class(
            mode != "api", "hidden")

    def _collect_semantic(self) -> SemanticSettings:
        settings = SemanticSettings(
            enabled=self.query_one("#semantic", Switch).value)
        for name, _label, _role in STAGES:
            settings.stages[name] = StageSettings(
                mode=self.query_one(f"#{name}-mode", Select).value,
                provider=self.query_one(f"#{name}-provider", Select).value,
                url=self.query_one(f"#{name}-url", Input).value,
                api_key=self.query_one(f"#{name}-key", Input).value,
                model=self.query_one(f"#{name}-model", Input).value,
                parallel=self.query_one(f"#{name}-parallel", Input).value,
            )
        return settings

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.exit(None)
            return
        if event.button.id != "start":
            return
        try:
            root, db, port = validate_setup(
                self.query_one("#root", Input).value,
                self.query_one("#db", Input).value,
                self.query_one("#port", Input).value,
            )
        except ValueError as exc:
            self.query_one("#error", Static).update(str(exc))
            return
        semantic = self._collect_semantic()
        db.parent.mkdir(parents=True, exist_ok=True)
        if semantic.enabled:
            try:
                validate_semantic(semantic, db.parent)
            except ValueError as exc:
                self.query_one("#error", Static).update(str(exc))
                return
            semantic.config_path = write_config(
                db.parent / CONFIG_NAME, semantic)
        self.exit((str(root), str(db), port, semantic))

    def action_cancel(self) -> None:
        self.exit(None)


def launch_setup(root: str = ".", db: str | None = None, port: int = 8765):
    """Run setup and return normalized values, or ``None`` when cancelled."""
    return SetupApp(root=root, db=db, port=port).run()
