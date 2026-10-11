"""Hardware-gated inference adapters (plan §3).

These thin wrappers import their accelerator runtimes lazily so the rest
of the package stays importable on any host. They are exercised only by
the opt-in MLX/ROCm smoke tests; absence of hardware/runtime raises a
clear error instead of silently using the CPU.
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_SUMMARY_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VULKAN_SUMMARY_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:*q8_0.gguf"
DEFAULT_VULKAN_EMBED_MODEL = "second-state/All-MiniLM-L6-v2-Embedding-GGUF:*Q8_0.gguf"


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


@lru_cache(maxsize=2)
def _mlx_summary_model(model_path: str):
    return _require("mlx_lm", "MLX").load(model_path)


def mlx_summary(prompt: str, max_output_tokens: int = 256,
                model_path: str = DEFAULT_SUMMARY_MODEL) -> str:
    mlx_lm = _require("mlx_lm", "MLX")
    model, tokenizer = _mlx_summary_model(model_path)
    return mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_output_tokens
    )


@lru_cache(maxsize=2)
def _mlx_embed_model(model_path: str):
    return _require("mlx_embeddings", "MLX").load(model_path)


def mlx_embed(texts, model_path: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    mlx_embeddings = _require("mlx_embeddings", "MLX")
    model, tokenizer = _mlx_embed_model(model_path)
    output = mlx_embeddings.generate(model, tokenizer, list(texts))
    # generate() returns a BaseModelOutput, not vectors: ``text_embeds`` is the
    # pooled sentence embedding; fall back to mean-pooling the token states.
    vectors = getattr(output, "text_embeds", None)
    if vectors is None:
        vectors = getattr(output, "last_hidden_state").mean(axis=1)
    return [_normalize(list(map(float, vector))) for vector in vectors.tolist()]


@lru_cache(maxsize=2)
def _torch_pipeline(model_path, device, backend):
    transformers = _require("transformers", backend)
    _require("torch", backend)
    return transformers.pipeline(
        "text-generation", model=model_path, device=device
    )


@lru_cache(maxsize=2)
def _torch_embed_model(model_path, device, backend):
    transformers = _require("transformers", backend)
    _require("torch", backend)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModel.from_pretrained(model_path)
    return model.to(device).eval(), tokenizer


def _torch_summary(prompt, max_output_tokens, model_path, device, backend):
    pipeline = _torch_pipeline(model_path, device, backend)
    result = pipeline(prompt, max_new_tokens=max_output_tokens,
                      return_full_text=False)
    return result[0]["generated_text"]


def _torch_embed(texts, model_path, device, backend):
    model, tokenizer = _torch_embed_model(model_path, device, backend)
    encoded = tokenizer(texts, padding=True, truncation=True,
                        return_tensors="pt").to(device)
    hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return [_normalize(vector.tolist()) for vector in pooled]


# PyTorch addresses ROCm GPUs as "cuda" too.
def rocm_summary(prompt: str, max_output_tokens: int = 256,
                 model_path: str = DEFAULT_SUMMARY_MODEL) -> str:
    return _torch_summary(prompt, max_output_tokens, model_path,
                          "cuda", "ROCm")


def rocm_embed(texts, model_path: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    return _torch_embed(texts, model_path, "cuda", "ROCm")


def cuda_summary(prompt: str, max_output_tokens: int = 256,
                 model_path: str = DEFAULT_SUMMARY_MODEL) -> str:
    return _torch_summary(prompt, max_output_tokens, model_path,
                          "cuda", "CUDA")


def cuda_embed(texts, model_path: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    return _torch_embed(texts, model_path, "cuda", "CUDA")


@lru_cache(maxsize=2)
def _llama_model(model_path: str, embedding: bool):
    """Load a GGUF model with every layer offloaded to the Vulkan device.

    ``model_path`` is a local ``.gguf`` file or ``repo_id:filename-glob``
    downloaded from Hugging Face.
    """
    llama_cpp = _require("llama_cpp", "Vulkan")
    options = dict(n_gpu_layers=-1, verbose=False, embedding=embedding,
                   n_ctx=0 if embedding else 4096)
    if model_path.endswith(".gguf"):
        return llama_cpp.Llama(model_path=model_path, **options)
    repo_id, _, filename = model_path.partition(":")
    return llama_cpp.Llama.from_pretrained(
        repo_id=repo_id, filename=filename or "*.gguf", **options)


def vulkan_summary(prompt: str, max_output_tokens: int = 256,
                   model_path: str = DEFAULT_VULKAN_SUMMARY_MODEL) -> str:
    result = _llama_model(model_path, False).create_completion(
        prompt, max_tokens=max_output_tokens)
    return result["choices"][0]["text"]


def vulkan_embed(texts, model_path: str = DEFAULT_VULKAN_EMBED_MODEL
                 ) -> list[list[float]]:
    response = _llama_model(model_path, True).create_embedding(list(texts))
    vectors = []
    for item in response["data"]:
        vector = item["embedding"]
        if vector and isinstance(vector[0], list):  # per-token: mean-pool
            vector = [sum(column) / len(vector) for column in zip(*vector)]
        vectors.append(_normalize(list(map(float, vector))))
    return vectors


@lru_cache(maxsize=2)
def _cpu_embed_model(model_path: str):
    transformers = _require("transformers", "CPU")
    _require("torch", "CPU")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModel.from_pretrained(model_path).eval()
    return model, tokenizer


def cpu_embed(texts, model_path: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    """Mean-pooled, normalized embeddings computed on the CPU."""
    torch = _require("torch", "CPU")
    model, tokenizer = _cpu_embed_model(model_path)
    encoded = tokenizer(texts, padding=True, truncation=True,
                        return_tensors="pt")
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return [_normalize(vector.tolist()) for vector in pooled]


def cpu_summary(prompt: str, max_output_tokens: int = 256,
                model_path: str = DEFAULT_SUMMARY_MODEL) -> str:
    pipeline = _torch_pipeline(model_path, "cpu", "CPU")
    result = pipeline(prompt, max_new_tokens=max_output_tokens,
                      return_full_text=False)
    return result[0]["generated_text"]
