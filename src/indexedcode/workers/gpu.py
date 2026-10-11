"""GPU vendor detection for backend selection (Linux).

Reads sysfs PCI vendor ids rather than shelling out, so it is cheap and
works inside containers. ``INDEXEDCODE_GPU`` (``nvidia``, ``amd`` or
``none``) overrides detection.
"""

from __future__ import annotations

import glob
import os
import shutil

NVIDIA, AMD, NONE = "nvidia", "amd", "none"
_VENDOR_IDS = {"0x10de": NVIDIA, "0x1002": AMD}


def detect_gpu(drm_glob: str = "/sys/class/drm/card*/device/vendor",
               which=shutil.which, environ=None) -> str:
    """``nvidia`` | ``amd`` | ``none``; NVIDIA wins on hybrid hosts."""
    environ = os.environ if environ is None else environ
    forced = environ.get("INDEXEDCODE_GPU", "").strip().lower()
    if forced in (NVIDIA, AMD, NONE):
        return forced
    vendors = set()
    for path in glob.glob(drm_glob):
        try:
            with open(path) as fh:
                vendors.add(_VENDOR_IDS.get(fh.read().strip().lower()))
        except OSError:
            continue
    if NVIDIA in vendors or which("nvidia-smi"):
        return NVIDIA
    if AMD in vendors:
        return AMD
    return NONE
