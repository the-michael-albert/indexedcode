"""CLI logging: human-readable lines on stderr.

stdout is reserved for machine output (JSON results, the MCP protocol), so
every log record goes to stderr. Only the ``indexedcode`` logger tree is
configured; third-party libraries keep their own defaults.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "indexedcode"
_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


def level_for(verbose: int = 0, quiet: bool = False) -> int:
    if quiet:
        return logging.WARNING
    return logging.DEBUG if verbose else logging.INFO


def configure_logging(verbose: int = 0, quiet: bool = False) -> None:
    """Send ``indexedcode`` log records to the current stderr. Idempotent."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_indexedcode_cli", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler._indexedcode_cli = True
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level_for(verbose, quiet))
    logger.propagate = False
