"""
Tests for the multi-provider LLM dispatch layer in ``jarvis.llm``.

The codebase historically talked to Ollama directly. These tests verify the
provider abstraction added so that setting ``llm_provider`` to ``anthropic``
routes chat calls to Anthropic's Messages API while still returning an
Ollama-shaped response to the rest of the codebase.

Embeddings always stay on Ollama regardless of provider (Anthropic does not
expose an embeddings API).
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from jarvis import llm as llm_module
from jarvis.llm import (
    call_llm_direct,
    call_llm_streaming,
    chat_with_messages,
    configure_llm_provider,
    extract_text_from_response,
)


# ---------------------------------------------------------------------------
# Fixtures — reset provider state between tests so leakage cannot mask bugs.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_provider_state():
    """Force the module-level provider config back to ollama after each test."""
    yield
    configure_llm_provider(provider="ollama")


def _set_anthropic(api_key: str = "sk-test", model: str = "claude-sonnet-4-6", max_tokens: int = 4096) -> None:
    configure_llm_provider(
        provider="anthropic",
        anthropic_api_key=api_key,
        anthropic_chat_model=model,
        anthropic_max_tokens=max_tokens,
    )


def _mock_anthropic_response(content_blocks, status_code: int = 200) -> MagicMock:
    """Build a MagicMock that mimics requests.Response for Anthropic /v1/messages."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        err = HTTPError()
        err.response = resp
        resp.raise_for_status.side_effect = err
    resp.json.return_value = {
        "id": "msg_01ABC",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    # context manager support
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.text = json.dumps(resp.json.return_value)
    return resp


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


class TestProviderRouting:
    """The default provider is ollama; explicit anthropic switches dispatch."""

    def test_default_provider_is_ollama(self):
        # Re-importing should leave provider at the safe default
        assert llm_module._PROVIDER_CONFIG["provider"] == "ollama"

    def test_configure_anthropic_sets_state(self):
        configure_llm_provider(
            provider="anthropic",
            anthropic_api_key="sk-test",
            anthropic_chat_model="claude-sonnet-4-6",
            anthropic_max_tokens=2048,
        )
        assert llm_module._PROVIDER_CONFIG["provider"] == "anthropic"
        assert llm_module._PROVIDER_CONFIG["anthropic_api_key"] == "sk-test"
        assert llm_module._PROVIDER_CONFIG["anthropic_chat_model"] == "claude-sonnet-4-6"
        assert llm_module._PROVIDER_CONFIG["anthropic_max_tokens"] == 2048

    def test_unknown_provider_falls_back_to_ollama(self):
        configure_llm_provider(provider="cohere")
        assert llm_module._PROVIDER_CONFIG["provider"] == "ollama"

    @patch("jarvis.llm.requests.post")
    def test_ollama_path_still_works_with_default_config(self, mock_post):
        resp = MagicMock()
        resp.json.return_value = {"message": {"content": "hello", "role": "assistant"}}
        resp.raise_for_status = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_post.return_value = resp

        out = call_llm_direct("http://127.0.0.1:11434", "gemma4:e2b", "sys", "hi")
        assert out == "hello"
        # Confirm we hit the Ollama endpoint, not Anthropic
        called_url = mock_post.call_args.args[0]
        assert "/api/chat" in called_url


# ---------------------------------------------------------------------------
# Anthropic chat — payload construction
# ---------------------------------------------------------------------------


class TestAnthropicPayload:
    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_hits_anthropic_endpoint(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response(
            [{"type": "text", "text": "Hi there."}]
        )

        out = call_llm_direct("http://ignored", "ignored-model", "You are X.", "Hello?")

        assert out == "Hi there."
        url = mock_post.call_args.args[0]
        assert "anthropic.com" in url
        assert url.endswith("/messages")

    @patch("jarvis.llm.requests.post")
    def test_anthropic_payload_extracts_system_to_top_level(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        call_llm_direct("http://ignored", "ignored", "SYS-PROMPT", "user-text")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["system"] == "SYS-PROMPT"
        # system must NOT appear inside messages
        assert all(m.get("role") != "system" for m in payload["messages"])

    @patch("jarvis.llm.requests.post")
    def test_anthropic_payload_uses_configured_model_not_caller_model(self, mock_post):
        _set_anthropic(model="claude-sonnet-4-6")
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        # Caller passes an Ollama model id — should be ignored.
        call_llm_direct("http://ignored", "gemma4:e2b", "sys", "hi")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "claude-sonnet-4-6"

    @patch("jarvis.llm.requests.post")
    def test_anthropic_payload_includes_max_tokens(self, mock_post):
        _set_anthropic(max_tokens=1234)
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        call_llm_direct("http://ignored", "ignored", "sys", "hi")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["max_tokens"] == 1234

    @patch("jarvis.llm.requests.post")
    def test_anthropic_headers_include_api_key_and_version(self, mock_post):
        _set_anthropic(api_key="sk-zzz")
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        call_llm_direct("http://ignored", "ignored", "sys", "hi")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "sk-zzz"
        assert "anthropic-version" in headers

    @patch("jarvis.llm.requests.post")
    def test_anthropic_call_fails_without_api_key(self, mock_post):
        configure_llm_provider(provider="anthropic", anthropic_api_key="")
        out = call_llm_direct("http://ignored", "ignored", "sys", "hi")
        assert out is None
        # The HTTP layer must not have been touched.
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Anthropic — message and tool translation
# ---------------------------------------------------------------------------


class TestAnthropicMessageTranslation:
    @patch("jarvis.llm.requests.post")
    def test_tool_role_becomes_tool_result_block(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what is the weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "getWeather", "arguments": {"city": "Tbilisi"}}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 24C"},
        ]
        chat_with_messages("http://ignored", "ignored", messages)

        payload = mock_post.call_args.kwargs["json"]
        # Tool result lives as a user message with a tool_result block.
        last = payload["messages"][-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "tool_result"
        assert last["content"][0]["tool_use_id"] == "call_1"
        assert "Sunny" in last["content"][0]["content"]

    @patch("jarvis.llm.requests.post")
    def test_assistant_tool_calls_become_tool_use_blocks(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "call_xyz", "type": "function", "function": {"name": "getWeather", "arguments": {"city": "Tbilisi"}}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_xyz", "content": "Sunny"},
        ]
        chat_with_messages("http://ignored", "ignored", messages)

        payload = mock_post.call_args.kwargs["json"]
        assistant_msg = payload["messages"][1]
        assert assistant_msg["role"] == "assistant"
        blocks = assistant_msg["content"]
        assert isinstance(blocks, list)
        # Text block + tool_use block
        types = [b["type"] for b in blocks]
        assert "tool_use" in types
        tu = [b for b in blocks if b["type"] == "tool_use"][0]
        assert tu["name"] == "getWeather"
        assert tu["input"] == {"city": "Tbilisi"}
        assert tu["id"] == "call_xyz"

    @patch("jarvis.llm.requests.post")
    def test_tool_call_arguments_parse_string_json(self, mock_post):
        """When an assistant tool_call carries arguments as a JSON string (the
        shape we send back to ourselves through Ollama responses), it must
        still be parsed into a dict for Anthropic's tool_use.input."""
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "getWeather", "arguments": '{"city": "Tbilisi"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        chat_with_messages("http://ignored", "ignored", messages)

        payload = mock_post.call_args.kwargs["json"]
        tu = [b for b in payload["messages"][1]["content"] if b["type"] == "tool_use"][0]
        assert tu["input"] == {"city": "Tbilisi"}


def _mock_anthropic_error(status_code: int, error_type: str, message: str) -> MagicMock:
    """Build a requests.Response-like mock for an Anthropic error response."""
    resp = MagicMock()
    resp.status_code = status_code
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    resp.json.return_value = body
    resp.text = json.dumps(body)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestAnthropicErrorSurfacing:
    """A failed Anthropic call must surface the API's own explanation, not an
    opaque '400 Bad Request'. Without the body the real cause (out-of-range
    field, inaccessible model, malformed block) is invisible to the user."""

    @patch("jarvis.llm.requests.post")
    def test_non_tools_400_surfaces_message_and_returns_none(self, mock_post, capsys):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_error(
            400, "invalid_request_error",
            "max_tokens: 200000 > 64000, which is the maximum allowed",
        )

        out = chat_with_messages("http://ignored", "ignored", [{"role": "user", "content": "hi"}])

        assert out is None
        printed = capsys.readouterr().out
        assert "max_tokens" in printed
        assert "64000" in printed

    @patch("jarvis.llm.requests.post")
    def test_base_payload_400_with_tools_is_not_misclassified_as_tools_error(self, mock_post, capsys):
        """A 400 whose message does not concern tools must NOT raise
        ToolsNotSupportedError just because tools were attached — that hid the
        real cause and permanently dropped native tool calling."""
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_error(
            400, "invalid_request_error", "max_tokens: must be greater than 0",
        )

        tools = [{"type": "function", "function": {"name": "x", "description": "", "parameters": {}}}]
        # Must not raise; must return None and surface the reason.
        out = chat_with_messages("http://ignored", "ignored", [{"role": "user", "content": "hi"}], tools=tools)
        assert out is None
        assert "max_tokens" in capsys.readouterr().out

    @patch("jarvis.llm.requests.post")
    def test_tools_schema_400_still_raises_tools_not_supported(self, mock_post):
        """When the 400 actually names the tools/schema, preserve the
        text-mode fallback signal the engine relies on."""
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_error(
            400, "invalid_request_error",
            "tools.0.input_schema: invalid JSON schema",
        )

        tools = [{"type": "function", "function": {"name": "x", "description": "", "parameters": {}}}]
        with pytest.raises(llm_module.ToolsNotSupportedError):
            chat_with_messages("http://ignored", "ignored", [{"role": "user", "content": "hi"}], tools=tools)

    @patch("jarvis.llm.requests.post")
    def test_401_surfaces_authentication_error(self, mock_post, capsys):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_error(401, "authentication_error", "invalid x-api-key")

        out = call_llm_direct("http://ignored", "ignored", "sys", "hi")
        assert out is None
        assert "authentication_error" in capsys.readouterr().out


def _assert_no_orphan_tool_use(messages):
    """Encode Anthropic's hard constraint: every tool_use block must be
    immediately followed by a user message containing a tool_result with the
    matching id. An orphan here is exactly what produced the production 400s."""
    for i, m in enumerate(messages):
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, list):
            tool_ids = [b["id"] for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            if not tool_ids:
                continue
            assert i + 1 < len(messages), f"tool_use at index {i} has no following message"
            nxt = messages[i + 1]
            assert nxt.get("role") == "user", f"tool_use must be followed by a user turn, got {nxt.get('role')!r}"
            ncontent = nxt.get("content")
            assert isinstance(ncontent, list), "tool_use must be followed by tool_result blocks, not text"
            result_ids = [b.get("tool_use_id") for b in ncontent if isinstance(b, dict) and b.get("type") == "tool_result"]
            for tid in tool_ids:
                assert tid in result_ids, f"tool_use id {tid!r} has no matching tool_result"


class TestAnthropicToolUsePairing:
    """Regression for the production 400: once a tool ran, the assistant
    tool_use was followed by a plain '[Tool result]' user message (the Ollama
    convention) rather than a tool_result block, orphaning the tool_use and
    making every subsequent Anthropic request fail."""

    def test_direct_exec_user_text_result_is_paired_as_tool_result(self):
        # Mirrors the planner direct-exec injection in engine.py.
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what's the temperature in Montreal"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_plan_abc123", "type": "function",
                     "function": {"name": "getWeather", "arguments": {"location": "Montreal"}}}
                ],
            },
            {"role": "user", "content": "[Tool result: getWeather]\nOvercast, 20.0C",
             "tool_name": "getWeather", "tool_failed": False},
            {"role": "user", "content": "how are you"},
        ]
        _system, out = llm_module._messages_to_anthropic(messages)
        _assert_no_orphan_tool_use(out)
        # The tool_use is paired to a tool_result that preserves the result text.
        tu = next(b for msg in out if isinstance(msg.get("content"), list)
                  for b in msg["content"] if b.get("type") == "tool_use")
        tr = next(b for msg in out if isinstance(msg.get("content"), list)
                  for b in msg["content"] if b.get("type") == "tool_result")
        assert tr["tool_use_id"] == tu["id"] == "call_plan_abc123"
        assert "Overcast" in tr["content"]

    def test_full_session_with_prior_tool_history_has_no_orphans(self):
        # The exact shape that 400'd in the field: a clean turn, then a tool
        # turn, then a follow-up question carrying the tool history.
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what time is it"},
            {"role": "assistant", "content": "It's 2:48 PM."},
            {"role": "user", "content": "temperature in Montreal"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_plan_x", "type": "function",
                             "function": {"name": "getWeather", "arguments": {"location": "Montreal"}}}]},
            {"role": "user", "content": "[Tool result: getWeather]\n20C", "tool_name": "getWeather"},
            {"role": "assistant", "content": "It's 20C in Montreal."},
            {"role": "user", "content": "how are you"},
        ]
        _system, out = llm_module._messages_to_anthropic(messages)
        _assert_no_orphan_tool_use(out)

    def test_trailing_tool_use_gets_synthetic_result(self):
        messages = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "getWeather", "arguments": {}}}]},
        ]
        _system, out = llm_module._messages_to_anthropic(messages)
        _assert_no_orphan_tool_use(out)

    def test_native_tool_role_still_pairs(self):
        # The pre-existing role="tool" path must keep working.
        messages = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "getWeather", "arguments": {"city": "Tbilisi"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
        ]
        _system, out = llm_module._messages_to_anthropic(messages)
        _assert_no_orphan_tool_use(out)


class TestAnthropicToolSchema:
    @patch("jarvis.llm.requests.post")
    def test_openai_tools_translated_to_anthropic_input_schema(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response([{"type": "text", "text": "ok"}])

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "getWeather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        chat_with_messages(
            "http://ignored", "ignored",
            [{"role": "user", "content": "weather?"}],
            tools=tools,
        )

        payload = mock_post.call_args.kwargs["json"]
        assert "tools" in payload
        t = payload["tools"][0]
        assert t["name"] == "getWeather"
        assert t["description"] == "Get weather for a city."
        # OpenAI's `parameters` becomes Anthropic's `input_schema`.
        assert t["input_schema"]["properties"]["city"]["type"] == "string"
        assert t["input_schema"]["required"] == ["city"]
        # Anthropic tool shape must not have OpenAI's `function` wrapper.
        assert "function" not in t
        assert "parameters" not in t


# ---------------------------------------------------------------------------
# Anthropic — response normalisation
# ---------------------------------------------------------------------------


class TestAnthropicResponseNormalisation:
    @patch("jarvis.llm.requests.post")
    def test_text_only_response_yields_ollama_shape(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response(
            [{"type": "text", "text": "Hello, world."}]
        )

        resp = chat_with_messages(
            "http://ignored", "ignored",
            [{"role": "user", "content": "hi"}],
        )

        assert isinstance(resp, dict)
        assert resp["message"]["role"] == "assistant"
        assert resp["message"]["content"] == "Hello, world."
        # No tool_calls in a pure text reply.
        assert "tool_calls" not in resp["message"] or not resp["message"]["tool_calls"]

    @patch("jarvis.llm.requests.post")
    def test_tool_use_response_maps_to_tool_calls(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response(
            [
                {"type": "text", "text": "Checking weather..."},
                {
                    "type": "tool_use",
                    "id": "toolu_01XYZ",
                    "name": "getWeather",
                    "input": {"city": "Tbilisi"},
                },
            ]
        )

        resp = chat_with_messages(
            "http://ignored", "ignored",
            [{"role": "user", "content": "weather"}],
            tools=[{"type": "function", "function": {"name": "getWeather", "description": "", "parameters": {}}}],
        )

        msg = resp["message"]
        # Text content is preserved (concatenated from text blocks).
        assert msg["content"] == "Checking weather..."
        # Tool call mapped to Ollama-shaped entry.
        tcs = msg["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["id"] == "toolu_01XYZ"
        assert tcs[0]["function"]["name"] == "getWeather"
        # Crucially, arguments come back as a dict (not a JSON string) so the
        # reply engine's _extract_structured_tool_call() can read them directly.
        assert tcs[0]["function"]["arguments"] == {"city": "Tbilisi"}

    @patch("jarvis.llm.requests.post")
    def test_extract_text_from_response_works_on_normalised_anthropic_shape(self, mock_post):
        _set_anthropic()
        mock_post.return_value = _mock_anthropic_response(
            [{"type": "text", "text": "Hi."}]
        )

        resp = chat_with_messages("http://ignored", "ignored", [{"role": "user", "content": "x"}])
        # The cross-codebase helper must still extract text out of the
        # normalised response.
        assert extract_text_from_response(resp) == "Hi."


# ---------------------------------------------------------------------------
# Anthropic — streaming
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:
    @patch("jarvis.llm.requests.post")
    def test_streaming_accumulates_text_delta_events_and_invokes_callback(self, mock_post):
        _set_anthropic()

        def _sse_lines():
            yield "event: message_start"
            yield 'data: {"type": "message_start"}'
            yield ""
            yield "event: content_block_delta"
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}}'
            yield ""
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}'
            yield ""
            yield 'data: {"type": "message_stop"}'
            yield ""

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = list(_sse_lines())
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_post.return_value = resp

        chunks: list[str] = []
        out = call_llm_streaming(
            "http://ignored", "ignored", "sys", "hi",
            on_token=lambda t: chunks.append(t),
        )

        assert out == "Hello"
        assert chunks == ["Hel", "lo"]
        # And the request must have been a stream request.
        assert mock_post.call_args.kwargs.get("stream") is True
        payload = mock_post.call_args.kwargs["json"]
        assert payload["stream"] is True
