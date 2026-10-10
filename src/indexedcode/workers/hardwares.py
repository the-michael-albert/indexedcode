"""Hardware-gated inference adapters (plan §3).

These thin wrappers import their accelerator runtimes lazily so the rest
of the package stays importable on any host. They are exercised only by
the opt-in MLX/ROCm smoke tests; absence of hardware/runtime raises a
clear error instead of silently using the CPU.
"""

from __future__ import annotations

DEFAULT_SUMMARY_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _require(import_name: str, backend: str):
    try:
        return __import__(import_name)
    except ImportError as exc:  # pragma: no cover - hardware gated
        raise RuntimeError(
            f"{backend} runtime {import_name!r} is not installed;"
            " refusing to fall back to another backend"
        ) from exc


def _normalize(vector) -> list[float]:
    norm = sum(component * component for component in vector) ** 0.5
    return [component / norm for component in vector]


def mlx_summary(prompt: str, max_output_tokens: int = 256,
                model_path: str = DEFAULT_SUMMARY_MODEL) -> str:
    mlx_lm = _require("mlx_lm", "MLX")
    model, tokenizer = mlx_lm.load(model_path)
    return mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_output_tokens
    )


def mlx_embed(texts, model_path: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    mlx_embeddings = _require("mlx_embeddings", "MLX")
    model, tokenizer = mlx_embeddings.load(model_path)
    outputs = mlx_embeddings.generate(model, tokenizer, texts)
    return [_normalize(list(map(float, vector))) for vector in outputs]


def rocm_summary(prompt: str, max_output_tokens: int = 256,
                 model_path: str = DEFAULT_SUMMARY_MODEL) -> str:
    transformers = _require("transformers", "ROCm")
    _require("torch", "ROCm")
    pipeline = transformers.pipeline(
        "text-generation", model=model_path, device="cuda"
    )
    result = pipeline(prompt, max_new_tokens=max_output_tokens)
    return result[0]["generated_text"]


def rocm_embed(texts, model_path: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    transformers = _require("transformers", "ROCm")
    _require("torch", "ROCm")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModel.from_pretrained(model_path)
    model = model.to("cuda").eval()
    encoded = tokenizer(texts, padding=True, truncation=True,
                        return_tensors="pt").to("cuda")
    hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return [_normalize(vector.tolist()) for vector in pooled]
