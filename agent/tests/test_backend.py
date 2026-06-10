"""Tests for the OpenAI-compatible VLM backend (no network)."""

from __future__ import annotations

import base64

import pytest

from swift_agent.backend import (
    BackendConfig,
    VLMBackend,
    build_user_content,
    encode_image_to_data_url,
)


# ---------------------------------------------------------------------------
# content building
# ---------------------------------------------------------------------------


def test_build_user_content_text_only_is_plain_string() -> None:
    assert build_user_content("hello", None) == "hello"


def test_build_user_content_with_image_is_parts_list(tmp_path) -> None:
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake png bytes")
    content = build_user_content("describe", img)
    # image part first, then text — mirrors io_builder convention.
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "describe"}


def test_encode_image_round_trips_base64(tmp_path) -> None:
    img = tmp_path / "y.jpg"
    payload = b"\xff\xd8\xff some jpeg bytes \x00\x01"
    img.write_bytes(payload)
    url = encode_image_to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == payload


def test_encode_image_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        encode_image_to_data_url(tmp_path / "nope.png")


# ---------------------------------------------------------------------------
# chat() request shaping + response parsing (mocked transport)
# ---------------------------------------------------------------------------


def _capture_post(captured: dict):
    def _post(url, headers, body, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout"] = timeout
        return {
            "choices": [{"message": {"content": "hi there"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }
    return _post


def test_chat_builds_correct_request_and_parses_usage() -> None:
    captured: dict = {}
    backend = VLMBackend(
        BackendConfig(base_url="http://h:9/v1", model="Qwen2.5-VL-7B-Instruct",
                      temperature=0.3, top_p=0.8, max_tokens=42),
        post_fn=_capture_post(captured),
    )
    resp = backend.chat([{"role": "user", "content": "yo"}])

    # request shaping
    assert captured["url"] == "http://h:9/v1/chat/completions"
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert captured["body"]["model"] == "Qwen2.5-VL-7B-Instruct"
    assert captured["body"]["temperature"] == 0.3
    assert captured["body"]["top_p"] == 0.8
    assert captured["body"]["max_tokens"] == 42
    assert captured["body"]["messages"] == [{"role": "user", "content": "yo"}]

    # response parsing
    assert resp.text == "hi there"
    assert resp.prompt_tokens == 11
    assert resp.completion_tokens == 3


def test_chat_malformed_response_raises() -> None:
    backend = VLMBackend(post_fn=lambda *a: {"unexpected": "shape"})
    with pytest.raises(ValueError, match="Malformed chat-completions response"):
        backend.chat([{"role": "user", "content": "x"}])


def test_chat_missing_usage_defaults_to_zero() -> None:
    backend = VLMBackend(
        post_fn=lambda *a: {"choices": [{"message": {"content": "ok"}}]},
    )
    resp = backend.chat([{"role": "user", "content": "x"}])
    assert resp.prompt_tokens == 0 and resp.completion_tokens == 0


def test_resolved_api_key_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert BackendConfig(api_key=None).resolved_api_key() == "EMPTY"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xyz")
    assert BackendConfig(api_key=None).resolved_api_key() == "sk-xyz"
    assert BackendConfig(api_key="explicit").resolved_api_key() == "explicit"
