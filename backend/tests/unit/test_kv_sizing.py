import pytest

from cachepilot.core.config import Settings
from cachepilot.kv.sizing import PRESETS, KVShape, bytes_per_token, describe


def test_llama2_7b_matches_the_default_setting() -> None:
    assert bytes_per_token("llama-2-7b") == 524_288 == Settings().kv_bytes_per_token


def test_grouped_query_attention_shrinks_kv_by_the_head_ratio() -> None:
    # Same layers and head_dim; 8 KV heads instead of 32.
    assert bytes_per_token("llama-3-8b") == bytes_per_token("llama-2-7b") // 4 == 128 * 1024


def test_known_shapes() -> None:
    assert bytes_per_token("qwen2.5-0.5b") == 12 * 1024
    assert bytes_per_token("mistral-7b") == 128 * 1024
    assert bytes_per_token("llama-3-70b") == 320 * 1024
    assert bytes_per_token("deepseek-v3") == 2 * 61 * 576 * 2


def test_fp8_halves_it() -> None:
    fp16 = PRESETS["llama-3-8b"]
    fp8 = KVShape("llama-3-8b-fp8", fp16.layers, fp16.kv_heads, fp16.head_dim, dtype_bytes=1)
    assert fp8.bytes_per_token == fp16.bytes_per_token // 2


def test_tokens_that_fit_and_describe() -> None:
    gib = 1024**3
    assert PRESETS["llama-2-7b"].tokens_that_fit(2 * gib) == 4096
    text = describe("llama-2-7b", kv_bytes=2 * gib, prompt_tokens=4096)
    assert "512 KiB/token" in text and "fits 1 of them" in text


def test_unknown_preset() -> None:
    with pytest.raises(ValueError, match="unknown KV preset"):
        bytes_per_token("gpt-oss-1t")


def test_settings_preset_overrides_bytes_per_token() -> None:
    settings = Settings(kv_model_preset="llama-3-8b")
    assert settings.effective_kv_bytes_per_token() == 128 * 1024
    assert Settings().effective_kv_bytes_per_token() == 524_288
