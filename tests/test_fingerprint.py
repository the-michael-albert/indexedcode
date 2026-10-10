"""Model fingerprint contract (plan §4.4).

The orchestrator validates worker fingerprints before using results; a
changed embedding fingerprint/dimension requires an explicit rebuild.
"""

from indexedcode.workers.fingerprint import model_fingerprint

BASE = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "rev-1",
    "backend": "mlx",
    "dimension": 384,
}


def test_fingerprint_is_deterministic_and_sha256_prefixed():
    first = model_fingerprint(**BASE)
    second = model_fingerprint(**BASE)
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_fingerprint_ignores_key_order():
    reversed_kwargs = dict(reversed(list(BASE.items())))
    assert model_fingerprint(**BASE) == model_fingerprint(**reversed_kwargs)


def test_fingerprint_stable_across_extra_metadata_ordering():
    a = model_fingerprint(**BASE, pooling="mean", normalized=True)
    b = model_fingerprint(normalized=True, **BASE, pooling="mean")
    assert a == b


def test_different_inputs_produce_different_fingerprints():
    reference = model_fingerprint(**BASE)
    variants = [
        {**BASE, "model": "other/model"},
        {**BASE, "revision": "rev-2"},
        {**BASE, "backend": "rocm"},
        {**BASE, "dimension": 768},
        {**BASE, "quantization": "4bit"},
    ]
    for variant in variants:
        assert model_fingerprint(**variant) != reference


def test_fingerprint_is_not_a_trivial_hash_of_concatenation():
    # collisions from naive string concatenation: "ab"+"c" == "a"+"bc"
    a = model_fingerprint(model="ab", revision="c")
    b = model_fingerprint(model="a", revision="bc")
    assert a != b
