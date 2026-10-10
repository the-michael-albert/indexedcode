"""Interactive Textual setup for starting a repository watcher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, Label, Static


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


class SetupApp(App[tuple[str, str, int] | None]):
    """Collect a repository, database location, and HTTP port."""

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
    Button { margin-left: 1; }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, root: str = ".", db: str | None = None, port: int = 8765):
        super().__init__()
        resolved = Path(root).expanduser().resolve()
        self.initial_root = resolved
        self.initial_db = Path(db).expanduser() if db else default_db_path(resolved)
        self.initial_port = port

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
        db.parent.mkdir(parents=True, exist_ok=True)
        self.exit((str(root), str(db), port))

    def action_cancel(self) -> None:
        self.exit(None)


def launch_setup(root: str = ".", db: str | None = None, port: int = 8765):
    """Run setup and return normalized values, or ``None`` when cancelled."""
    return SetupApp(root=root, db=db, port=port).run()
