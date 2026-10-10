"""Exact model fingerprints (plan §4.4).

The orchestrator validates worker fingerprints before using results; a
changed embedding fingerprint/dimension requires an explicit rebuild into
a new database. Fingerprints hash a canonical JSON encoding so key order
never matters and naive string concatenation cannot collide.
"""

from __future__ import annotations

import hashlib
import json


def model_fingerprint(**details) -> str:
    """Return the ``sha256:...`` fingerprint of a model configuration."""
    canonical = json.dumps(details, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
