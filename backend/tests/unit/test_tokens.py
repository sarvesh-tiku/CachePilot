from cachepilot.core.models import Message
from cachepilot.core.tokens import estimate_prompt_tokens, estimate_text_tokens, normalize_prompt


def test_text_tokens_rounds_up() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abc") == 1
    assert estimate_text_tokens("a" * 8) == 2
    assert estimate_text_tokens("a" * 9) == 3


def test_normalize_prompt_is_role_prefixed_and_newline_terminated() -> None:
    messages = [Message(role="system", content="S"), Message(role="user", content="U")]
    assert normalize_prompt(messages) == "system:S\nuser:U\n"


def test_prompt_tokens_derive_from_normalized_text() -> None:
    messages = [Message(role="system", content="a" * 40), Message(role="user", content="b" * 4)]
    assert estimate_prompt_tokens(messages) == estimate_text_tokens(
        "system:" + "a" * 40 + "\n" + "user:" + "b" * 4 + "\n"
    )


def test_role_changes_estimate_input() -> None:
    as_user = [Message(role="user", content="x")]
    as_system = [Message(role="system", content="x")]
    assert normalize_prompt(as_user) != normalize_prompt(as_system)
