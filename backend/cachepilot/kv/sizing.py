from __future__ import annotations

from dataclasses import dataclass

KIB = 1024


@dataclass(frozen=True)
class KVShape:
    """What determines KV-cache bytes per token for a decoder-only transformer.

    Every layer stores one K and one V vector per token for each *KV* head
    (grouped-query attention shares KV heads across query heads, which is why
    Llama 3 8B needs a quarter of Llama 2 7B's KV memory at the same size).
    Multi-head latent attention (DeepSeek V2/V3) instead stores one compressed
    latent per token per layer; model it with kv_heads=1 and head_dim set to
    the latent width.
    """

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16 / bf16; 1 for fp8 KV cache

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    def tokens_that_fit(self, kv_bytes: int) -> int:
        return kv_bytes // self.bytes_per_token if self.bytes_per_token else 0


PRESETS: dict[str, KVShape] = {
    shape.name: shape
    for shape in (
        KVShape("llama-2-7b", layers=32, kv_heads=32, head_dim=128),
        KVShape("llama-3-8b", layers=32, kv_heads=8, head_dim=128),
        KVShape("llama-3-70b", layers=80, kv_heads=8, head_dim=128),
        KVShape("mistral-7b", layers=32, kv_heads=8, head_dim=128),
        KVShape("qwen2.5-0.5b", layers=24, kv_heads=2, head_dim=64),
        KVShape("qwen2.5-7b", layers=28, kv_heads=4, head_dim=128),
        KVShape("qwen2.5-72b", layers=80, kv_heads=8, head_dim=128),
        # MLA: 512-dim compressed KV latent + 64-dim decoupled RoPE key per token per layer.
        KVShape("deepseek-v3", layers=61, kv_heads=1, head_dim=512 + 64),
    )
}


def bytes_per_token(preset: str) -> int:
    try:
        return PRESETS[preset].bytes_per_token
    except KeyError:
        raise ValueError(f"unknown KV preset {preset!r}; known: {sorted(PRESETS)}") from None


def describe(preset: str, *, kv_bytes: int, prompt_tokens: int) -> str:
    shape = PRESETS[preset]
    per_token = shape.bytes_per_token
    return (
        f"{shape.name}: {per_token / KIB:.0f} KiB/token "
        f"({shape.layers} layers x {shape.kv_heads} KV heads x {shape.head_dim} dims x "
        f"{shape.dtype_bytes} B x K,V); a {prompt_tokens}-token prompt holds "
        f"{prompt_tokens * per_token / KIB / KIB:.0f} MiB and {kv_bytes / KIB / KIB / KIB:.0f} GiB "
        f"fits {shape.tokens_that_fit(kv_bytes) // prompt_tokens} of them"
    )
