#!/usr/bin/env python
"""Build the indexing-test fixture corpus and zip it (thin wrapper)."""

from __future__ import annotations

import sys
from pathlib import Path

from indexedcode.fixture import write_zip

if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1
                  else Path(__file__).parent / "indexing-fixture.zip")
    print(f"wrote {write_zip(target)}")
