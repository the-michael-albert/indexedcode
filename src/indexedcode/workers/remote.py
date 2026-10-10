"""Remote worker handle (plan §4.1).

Remote mode never spawns a process: the remote worker's host owns its
supervision. The orchestrator only holds an authenticated HTTP client.
"""

from __future__ import annotations

from indexedcode.workers.client import WorkerClient


class RemoteWorker:
    """A configured remote inference endpoint (no process supervision)."""

    def __init__(self, name: str, client: WorkerClient):
        self.name = name
        self.client = client

    def status(self) -> dict:
        return {
            "name": self.name,
            "mode": "remote",
            "base_url": self.client.base_url,
            "expected_fingerprint": self.client.expected_fingerprint,
        }
