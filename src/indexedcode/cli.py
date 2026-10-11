"""Command-line adapters (plan §6).

``indexedcode orchestrator serve`` — the normal local entry point:
writer, watcher, pipeline loop, and retrieval HTTP API in one process.
``indexedcode index once`` — one-shot reconcile-and-process. Query
commands print bounded JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time

from indexedcode import retrieval
from indexedcode.db import WriterLock, connect, initialize
from indexedcode.logs import configure_logging
from indexedcode.pipeline import run_until_idle
from indexedcode.reconcile import reconcile

logger = logging.getLogger(__name__)


def _reconcile_logged(conn, root, what):
    """reconcile() plus one log line when it found work (or on request)."""
    result = reconcile(conn, root, now_ms=int(time.time() * 1000))
    if result.refresh_enqueued or result.remove_enqueued:
        logger.info("%s: %d to index, %d to remove, %d unchanged", what,
                    result.refresh_enqueued, result.remove_enqueued,
                    result.unchanged)
    else:
        logger.debug("%s: nothing changed (%d unchanged, %d skipped)", what,
                     result.unchanged, result.skipped)
    logger.info("%s: scanned %d candidate files, %d skipped", what,
                result.candidates, result.skipped)
    return result


def _open_writer(db_path):
    conn = connect(db_path, writer=True, threaded=True)
    initialize(conn)
    return conn


def _cmd_index_once(args) -> int:
    conn = _open_writer(args.db)
    try:
        with WriterLock(_writer_lock_path(args.db)):
            logger.info("indexing %s", args.root)
            result = reconcile(conn, args.root)
            logger.info("found %d to index, %d to remove, %d unchanged "
                        "(%d candidates, %d skipped)",
                        result.refresh_enqueued, result.remove_enqueued,
                        result.unchanged, result.candidates, result.skipped)
            counts = run_until_idle(conn, args.root,
                                    now_ms=int(time.time() * 1000))
            logger.info("done: %d indexed, %d removed, %d retried, %d failed",
                        counts.get("completed", 0), counts.get("removed", 0),
                        counts.get("retried", 0), counts.get("failed", 0))
            ready = conn.execute(
                "SELECT COUNT(*) AS c FROM files WHERE status = 'ready'"
            ).fetchone()["c"]
        print(json.dumps({
            "files_ready": ready,
            "files_refreshed": result.refresh_enqueued,
            "files_removed": result.remove_enqueued,
            "files_unchanged": result.unchanged,
            "files_skipped": result.skipped,
        }))
        return 0
    finally:
        conn.close()


def _cmd_search(args) -> int:
    conn = connect(args.db, threaded=True)
    try:
        result = retrieval.search(
            conn, args.root, args.query, mode=args.mode, limit=args.limit,
        )
        print(json.dumps(result))
        return 0
    finally:
        conn.close()


def _cmd_context(args) -> int:
    conn = connect(args.db, threaded=True)
    try:
        result = retrieval.get_context(
            conn, args.root, path=args.path,
            start_line=args.start_line, end_line=args.end_line,
        )
        print(json.dumps(result))
        return 0
    finally:
        conn.close()


def _cmd_references(args) -> int:
    conn = connect(args.db, threaded=True)
    try:
        result = retrieval.get_references(conn, args.root, args.symbol)
        print(json.dumps(result))
        return 0
    finally:
        conn.close()


def _cmd_status(args) -> int:
    conn = connect(args.db, threaded=True)
    try:
        print(json.dumps(retrieval.get_status(conn, args.root)))
        return 0
    finally:
        conn.close()


def _cmd_setup(args) -> int:
    from indexedcode.setup_ui import launch_setup

    selected = launch_setup(root=args.root, db=args.db, port=args.http_port)
    if selected is None:
        return 0
    args.root, args.db, args.http_port = selected
    print(f"watching {args.root}", file=sys.stderr, flush=True)
    return _cmd_serve(args)


def _cmd_serve(args) -> int:
    from indexedcode.http_api import create_server
    from indexedcode.watcher import EventWatcher

    stop = threading.Event()
    conn = _open_writer(args.db)

    with WriterLock(_writer_lock_path(args.db)):
        logger.info("watching %s", args.root)
        logger.info("index database %s", args.db)
        # initial reconciliation: index the tree before accepting traffic
        _reconcile_logged(conn, args.root, "initial scan")
        _drain(conn, args.root)
        logger.info("initial index ready")

        dirty = set()

        def on_flush(paths):
            dirty.update(paths)

        watcher = EventWatcher(on_flush=on_flush,
                               debounce_s=args.debounce_s)

        def watch_events():
            try:
                from watchfiles import watch

                for changes in watch(args.root, stop_event=stop,
                                     watch_filter=None):
                    for _kind, path in changes:
                        rel = _relative(path, args.root)
                        if rel:
                            watcher.record(rel, now=time.monotonic())
            except Exception as exc:
                # the periodic scan below keeps the index converging
                logger.warning("file watcher stopped (%s); falling back to"
                               " periodic scans", exc)

        threading.Thread(target=watch_events, daemon=True).start()

        httpd = create_server(conn, args.root, token=args.http_token,
                              port=args.http_port)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"listening {httpd.base_url}", file=sys.stderr, flush=True)
        logger.info("dashboard %s/  ·  agent docs %s/llms.txt",
                    httpd.base_url, httpd.base_url)
        if args.http_token is None:
            logger.debug("no --http-token: loopback-only access")

        def request_stop(signum, frame):
            logger.info("stopping (%s)", signal.Signals(signum).name)
            stop.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        last_reconcile = time.monotonic()
        try:
            while not stop.is_set():
                now = time.monotonic()
                watcher.flush(now)
                if dirty:
                    logger.debug("%d changed path(s) from the file watcher",
                                 len(dirty))
                    dirty.clear()
                    _reconcile_logged(conn, args.root, "change detected")
                _drain(conn, args.root)
                if now - last_reconcile >= args.poll_interval_s:
                    _reconcile_logged(conn, args.root, "periodic scan")
                    last_reconcile = now
                stop.wait(0.1)
        finally:
            httpd.shutdown()
            httpd.server_close()
    conn.close()
    logger.info("stopped")
    return 0


def _drain(conn, root):
    """Process every due job; the pipeline logs each file it handles."""
    return run_until_idle(conn, root, now_ms=int(time.time() * 1000))


def _writer_lock_path(db_path):
    return f"{db_path}.writerlock"


def _relative(path, root):
    from pathlib import Path, PurePath

    try:
        rel = Path(path).relative_to(Path(root))
    except ValueError:
        return None
    if rel.parts and ".." in rel.parts:
        return None
    return PurePath(*rel.parts).as_posix()


def build_parser():
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed_version = version("indexedcode")
    except PackageNotFoundError:
        installed_version = "unknown (not installed)"
    parser = argparse.ArgumentParser(prog="indexedcode")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {installed_version}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true",
                           help="also log debug detail (HTTP requests,"
                                " scan results)")
    verbosity.add_argument("-q", "--quiet", action="store_true",
                           help="log warnings and errors only")

    def add_common(sub):
        sub.add_argument("--root", default=".")
        sub.add_argument("--db", default="index.sqlite")

    sub = parser.add_subparsers(dest="group", required=True)

    setup = sub.add_parser(
        "setup", help="interactively choose and watch a repository",
    )
    setup.add_argument("--root", default=".")
    setup.add_argument("--db", default=None)
    setup.add_argument("--http-port", type=int, default=8765)
    setup.add_argument("--http-token", default=None)
    setup.add_argument("--debounce-s", type=float, default=0.25)
    setup.add_argument("--poll-interval-s", type=float, default=5.0)
    setup.set_defaults(func=_cmd_setup)

    orchestrator = sub.add_parser("orchestrator")
    serve = orchestrator.add_subparsers(dest="action", required=True)
    serve_parser = serve.add_parser("serve")
    add_common(serve_parser)
    serve_parser.add_argument("--http-port", type=int, default=8765)
    serve_parser.add_argument("--http-token", default=None)
    serve_parser.add_argument("--debounce-s", type=float, default=0.25)
    serve_parser.add_argument("--poll-interval-s", type=float, default=5.0)
    serve_parser.set_defaults(func=_cmd_serve)

    index = sub.add_parser("index")
    once = index.add_subparsers(dest="action", required=True)
    once_parser = once.add_parser("once")
    add_common(once_parser)
    once_parser.set_defaults(func=_cmd_index_once)

    search = sub.add_parser("search")
    search.add_argument("query")
    add_common(search)
    search.add_argument("--mode", default="hybrid")
    search.add_argument("--limit", type=int, default=5)
    search.set_defaults(func=_cmd_search)

    context = sub.add_parser("context")
    context.add_argument("path")
    add_common(context)
    context.add_argument("--start-line", type=int)
    context.add_argument("--end-line", type=int)
    context.set_defaults(func=_cmd_context)

    references = sub.add_parser("references")
    references.add_argument("symbol")
    add_common(references)
    references.set_defaults(func=_cmd_references)

    status = sub.add_parser("status")
    add_common(status)
    status.set_defaults(func=_cmd_status)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=int(args.verbose), quiet=args.quiet)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
