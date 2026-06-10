"""OpenAI-compatible VLM backend client.

Wraps a chat-completions endpoint serving Qwen2.5-VL (vLLM / TensorRT-LLM
/ sglang). Image observations are sent as standard OpenAI multimodal
``content`` parts (``{"type": "image_url", ...}`` + ``{"type": "text",
...}``), mirroring the chat-messages convention the team's speculative
benchmark already uses (`edge-deployment/benchmark/speculative/io_builder.py`).

The HTTP call goes through the ``requests`` library against the
``/v1/chat/completions`` route, so no vendor SDK is required. The
backend is deliberately thin: it builds the request, posts it, and
returns the assistant message text plus token-usage metadata.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import mimetypes
import os
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class BackendConfig:
    """Configuration for the OpenAI-compatible chat backend.

    Attributes:
        base_url: Root of the OpenAI-compatible server, e.g.
            ``http://127.0.0.1:8000/v1``. The ``/chat/completions`` path
            is appended automatically.
        model: Served model name (default mirrors the team's
            ``Qwen2.5-VL-7B-Instruct`` benchmark target).
        api_key: Bearer token; most local servers ignore it but the
            header is still sent. Falls back to the ``OPENAI_API_KEY``
            env var, then a dummy value.
        temperature / top_p / max_tokens: standard sampling knobs.
        timeout_s: per-request HTTP timeout.
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen2.5-VL-7B-Instruct"
    api_key: str | None = None
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512
    timeout_s: float = 120.0

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")


def encode_image_to_data_url(image_path: str | Path) -> str:
    """Read an image file and return a ``data:`` URL for the OpenAI API.

    Local servers accept inline base64 data URLs in the
    ``image_url`` content part, which avoids needing the server to fetch
    a remote URL or share a filesystem with the agent.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_user_content(text: str, image_path: str | Path | None) -> Any:
    """Build the OpenAI ``content`` field for a user turn.

    Text-only turns return a plain string; multimodal turns return the
    ``[image_url, text]`` list form. Matches the team's io_builder
    convention (image part first, then the text prompt).
    """
    if image_path is None:
        return text
    return [
        {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}},
        {"type": "text", "text": text},
    ]


@dataclasses.dataclass
class ChatResponse:
    """Parsed assistant reply + usage metadata."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] | None = None


class VLMBackend:
    """Thin OpenAI-compatible chat client for the served VLM.

    The ``_post`` method is the single network seam — tests inject a
    fake by subclassing or by passing a ``post_fn``. This keeps the
    whole agent testable without a GPU or a running server.
    """

    def __init__(
        self,
        config: BackendConfig | None = None,
        post_fn: Any = None,
    ) -> None:
        self.config = config or BackendConfig()
        # ``post_fn(url, headers, json_body, timeout) -> dict`` lets tests
        # substitute the transport. Defaults to a lazily-imported
        # requests.post wrapper so the import isn't required for tests.
        self._post_fn = post_fn

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.resolved_api_key()}",
        }
        if self._post_fn is not None:
            return self._post_fn(url, headers, body, self.config.timeout_s)

        import requests  # local import: tests don't need it

        resp = requests.post(
            url, headers=headers, data=json.dumps(body), timeout=self.config.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, Any]]) -> ChatResponse:
        """Send a full ``messages`` list and return the assistant reply.

        ``messages`` follows the OpenAI schema; callers typically build
        it via ``build_user_content`` for the multimodal user turn.
        """
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        data = self._post(body)
        return self._parse(data)

    @staticmethod
    def _parse(data: dict[str, Any]) -> ChatResponse:
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Malformed chat-completions response: {data!r}") from e
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            raw=data,
        )


__all__ = [
    "BackendConfig",
    "VLMBackend",
    "ChatResponse",
    "encode_image_to_data_url",
    "build_user_content",
]
