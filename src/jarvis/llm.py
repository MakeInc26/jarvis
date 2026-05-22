"""Direct LLM interaction utilities without extra features like temporal context.

Provider routing
----------------
This module owns the wire-level chat API. Historically it spoke only Ollama;
now it dispatches between two backends based on ``configure_llm_provider``:

- ``"ollama"`` (default) — POSTs to ``{base_url}/api/chat`` with the Ollama
  payload shape (``options.num_ctx``, ``think``, OpenAI-style ``tools``).
- ``"anthropic"`` — POSTs to ``{anthropic_base_url}/messages`` with
  ``x-api-key`` + ``anthropic-version`` headers, extracts the system slot,
  translates tools to ``input_schema`` form, and normalises the response
  back to Ollama's ``{message: {content, tool_calls}}`` shape so all
  downstream call sites stay agnostic.

In Anthropic mode the per-call ``chat_model`` argument is **ignored** — every
chat slot (intent judge, planner, evaluator, main reply) runs on
``anthropic_chat_model`` from the provider config. Embeddings always use
Ollama regardless of provider (see ``jarvis.memory.embeddings``).
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Tuple, Generator, Callable
import requests
import json

from .debug import debug_log


class ToolsNotSupportedError(Exception):
    """Raised when the model returns HTTP 400 because native tool calling is not supported."""
    pass


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Module-level provider state. Set once at daemon startup via
# ``configure_llm_provider``. Defaults keep behaviour identical to the
# pre-multi-provider codebase: Ollama, no API key needed.
_PROVIDER_CONFIG: Dict[str, Any] = {
    "provider": "ollama",
    "anthropic_api_key": "",
    "anthropic_chat_model": "claude-sonnet-4-6",
    "anthropic_max_tokens": 4096,
    "anthropic_base_url": "https://api.anthropic.com/v1",
    "anthropic_version": "2023-06-01",
}


def configure_llm_provider(
    provider: str = "ollama",
    anthropic_api_key: str = "",
    anthropic_chat_model: str = "claude-sonnet-4-6",
    anthropic_max_tokens: int = 4096,
    anthropic_base_url: str = "https://api.anthropic.com/v1",
) -> None:
    """Configure provider routing for all chat calls in this module.

    Called once from the daemon entry point after settings are loaded.
    Unknown providers fall back to ``"ollama"`` so a typo can never silently
    break chat. The API key is trimmed; an empty key in Anthropic mode is
    valid as a config state (e.g. before the user fills it in) and surfaces
    as a clean failure on the first chat call rather than at startup.
    """
    provider = (provider or "ollama").strip().lower()
    if provider not in ("ollama", "anthropic"):
        provider = "ollama"
    _PROVIDER_CONFIG["provider"] = provider
    _PROVIDER_CONFIG["anthropic_api_key"] = (anthropic_api_key or "").strip()
    _PROVIDER_CONFIG["anthropic_chat_model"] = (anthropic_chat_model or "claude-sonnet-4-6").strip()
    try:
        _PROVIDER_CONFIG["anthropic_max_tokens"] = int(anthropic_max_tokens) if anthropic_max_tokens else 4096
    except (TypeError, ValueError):
        _PROVIDER_CONFIG["anthropic_max_tokens"] = 4096
    _PROVIDER_CONFIG["anthropic_base_url"] = (anthropic_base_url or "https://api.anthropic.com/v1").rstrip("/")
    debug_log(f"LLM provider configured: {provider} (model={_PROVIDER_CONFIG['anthropic_chat_model'] if provider == 'anthropic' else 'see caller'})", "llm")


def _is_anthropic() -> bool:
    return _PROVIDER_CONFIG["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# Anthropic translation layer
# ---------------------------------------------------------------------------


def _anthropic_headers() -> Dict[str, str]:
    return {
        "x-api-key": _PROVIDER_CONFIG["anthropic_api_key"],
        "anthropic-version": _PROVIDER_CONFIG["anthropic_version"],
        "content-type": "application/json",
    }


def _messages_to_anthropic(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Split system messages out and convert Ollama-shaped tool turns.

    Returns ``(system_text, anthropic_messages)``. System messages are joined
    on double-newline and returned separately (Anthropic puts ``system`` at the
    top level, not in the messages array). ``role=tool`` messages become a
    user message wrapping a single ``tool_result`` block. Assistant messages
    that carry ``tool_calls`` get rewritten as a content list with optional
    leading text and one ``tool_use`` block per call.
    """
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        if role == "tool":
            tool_use_id = m.get("tool_call_id") or m.get("id") or ""
            text = content if isinstance(content, str) else json.dumps(content)
            out.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": str(tool_use_id), "content": text}
                ],
            })
            continue
        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                blocks: List[Dict[str, Any]] = []
                if isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    # Tool-call arguments can arrive as a dict (Ollama native
                    # tool API) or as a JSON-encoded string (some text-mode
                    # parsers serialise before re-injecting). Both must
                    # become a dict for Anthropic's tool_use.input.
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": str(tc.get("id") or ""),
                        "name": str(fn.get("name") or ""),
                        "input": args,
                    })
                out.append({"role": "assistant", "content": blocks})
                continue
        # Plain user/assistant message with string content.
        normalised_role = role if role in ("user", "assistant") else "user"
        out.append({"role": normalised_role, "content": content})
    return "\n\n".join(system_parts), out


def _tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate OpenAI-style tool definitions to Anthropic ``input_schema`` form."""
    result: List[Dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            result.append({
                "name": str(fn.get("name", "")),
                "description": str(fn.get("description", "")),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        elif "name" in t and "input_schema" in t:
            # Already in Anthropic shape (defensive — caller built it directly).
            result.append(t)
    return result


def _anthropic_to_ollama_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise an Anthropic Messages response to the Ollama chat shape.

    The reply engine's ``_extract_structured_tool_call`` reads
    ``resp["message"]["tool_calls"][0]["function"]["arguments"]`` as a dict.
    Anthropic returns ``tool_use`` blocks with ``input`` already as a dict, so
    we pass it through verbatim — no JSON re-serialise.
    """
    content_blocks = data.get("content") or []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text")
            if isinstance(t, str):
                text_parts.append(t)
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id") or "",
                "type": "function",
                "function": {
                    "name": block.get("name") or "",
                    "arguments": block.get("input") or {},
                },
            })
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message, "done": True}


def _anthropic_build_payload(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    thinking: bool,
    temperature: Optional[float],
    stream: bool,
) -> Dict[str, Any]:
    system_text, anth_messages = _messages_to_anthropic(messages)
    payload: Dict[str, Any] = {
        "model": _PROVIDER_CONFIG["anthropic_chat_model"],
        "max_tokens": int(_PROVIDER_CONFIG["anthropic_max_tokens"]),
        "messages": anth_messages,
    }
    if system_text:
        payload["system"] = system_text
    if tools:
        payload["tools"] = _tools_to_anthropic(tools)
    if thinking:
        # Anthropic requires a budget. Half of max_tokens is a sane starting
        # point that matches the rest of the codebase's "thinking is optional
        # extra reasoning, not the bulk of the response" intent.
        budget = max(1024, int(_PROVIDER_CONFIG["anthropic_max_tokens"]) // 2)
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if stream:
        payload["stream"] = True
    return payload


def _call_anthropic_chat(
    messages: List[Dict[str, Any]],
    timeout_sec: float,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
    temperature: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Make a non-streaming Anthropic Messages call and return Ollama-shaped response."""
    api_key = _PROVIDER_CONFIG["anthropic_api_key"]
    if not api_key:
        debug_log("anthropic: missing API key — set anthropic_api_key in config", "llm")
        print("  ❌ LLM error: anthropic_api_key not configured", flush=True)
        return None

    payload = _anthropic_build_payload(messages, tools, thinking, temperature, stream=False)
    url = f"{_PROVIDER_CONFIG['anthropic_base_url'].rstrip('/')}/messages"
    headers = _anthropic_headers()

    try:
        with requests.post(url, json=payload, headers=headers, timeout=timeout_sec) as resp:
            if resp.status_code == 400 and tools:
                # Surface a tool-incompatibility signal that mirrors the Ollama
                # path so the reply engine's text-mode fallback can kick in.
                raise ToolsNotSupportedError(
                    f"Anthropic API returned HTTP 400 with tools — {resp.text[:200] if hasattr(resp, 'text') else ''}"
                )
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict):
            return _anthropic_to_ollama_response(data)
    except ToolsNotSupportedError:
        raise
    except requests.exceptions.Timeout:
        debug_log(f"anthropic: timeout after {timeout_sec}s", "llm")
        print("  ⏱️ LLM request timed out", flush=True)
        return None
    except requests.exceptions.ConnectionError as e:
        debug_log(f"anthropic: connection error — {e}", "llm")
        print(f"  ❌ LLM connection error: {e}", flush=True)
        return None
    except Exception as e:
        debug_log(f"anthropic: request failed — {e}", "llm")
        print(f"  ❌ LLM error: {e}", flush=True)
        return None
    return None


def _call_anthropic_stream(
    messages: List[Dict[str, Any]],
    timeout_sec: float,
    on_token: Optional[Callable[[str], None]] = None,
    thinking: bool = False,
) -> Optional[str]:
    """Stream an Anthropic SSE response and return the accumulated text.

    Parses ``content_block_delta`` events of type ``text_delta`` and invokes
    ``on_token`` for each chunk, mirroring the Ollama streaming contract.
    """
    api_key = _PROVIDER_CONFIG["anthropic_api_key"]
    if not api_key:
        debug_log("anthropic stream: missing API key", "llm")
        return None

    payload = _anthropic_build_payload(messages, tools=None, thinking=thinking, temperature=None, stream=True)
    url = f"{_PROVIDER_CONFIG['anthropic_base_url'].rstrip('/')}/messages"
    headers = _anthropic_headers()

    text_parts: List[str] = []
    try:
        with requests.post(url, json=payload, headers=headers, timeout=timeout_sec, stream=True) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "content_block_delta":
                    delta = evt.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        chunk = delta.get("text", "")
                        if chunk:
                            text_parts.append(chunk)
                            if on_token:
                                try:
                                    on_token(chunk)
                                except Exception:
                                    pass
    except requests.exceptions.Timeout:
        debug_log(f"anthropic stream: timeout after {timeout_sec}s", "llm")
        return None
    except Exception as e:
        debug_log(f"anthropic stream: failed — {e}", "llm")
        return None

    return "".join(text_parts) if text_parts else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_llm_direct(base_url: str, chat_model: str, system_prompt: str, user_content: str, timeout_sec: float = 10.0, thinking: bool = False, num_ctx: int = 4096, temperature: Optional[float] = None) -> Optional[str]:
    """Direct LLM call without temporal context, location, or other ask_coach features.

    ``num_ctx`` controls Ollama's context window for this call. Default 4096 is
    fine for small classification-shaped passes; callers that assemble richer
    prompts (planner with dialogue + memory + tool catalogue) should pass a
    larger value to avoid silent truncation. Ignored when provider is
    ``"anthropic"`` — Anthropic uses ``max_tokens`` (from provider config).

    ``temperature`` is forwarded to both providers when set. Pass ``0.0`` for
    classification / extraction calls where determinism beats creativity.

    When provider is ``"anthropic"``, ``base_url`` and ``chat_model`` are
    ignored; the call goes to the configured Anthropic endpoint using
    ``anthropic_chat_model``.
    """
    if _is_anthropic():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = _call_anthropic_chat(messages, timeout_sec, thinking=thinking, temperature=temperature)
        if resp:
            content = extract_text_from_response(resp)
            if isinstance(content, str) and content.strip():
                return content
            debug_log(f"call_llm_direct (anthropic): empty content from response keys={list(resp.keys())}", "llm")
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    options: Dict[str, Any] = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": options,
        "think": thinking,
    }

    try:
        with requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec) as resp:
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, dict):
            content = extract_text_from_response(data)
            if isinstance(content, str) and content.strip():
                return content
            debug_log(f"call_llm_direct: empty content from response keys={list(data.keys())}", "llm")
    except requests.exceptions.Timeout:
        debug_log(f"call_llm_direct: timeout after {timeout_sec}s", "llm")
        return None
    except Exception as e:
        debug_log(f"call_llm_direct: request failed — {e}", "llm")
        return None

    return None


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
) -> Optional[str]:
    """
    Streaming LLM call that invokes on_token callback for each token received.

    Args:
        base_url: Ollama base URL (ignored when provider is "anthropic")
        chat_model: Model name (ignored when provider is "anthropic")
        system_prompt: System prompt
        user_content: User message
        on_token: Callback invoked with each token as it arrives
        timeout_sec: Request timeout
        thinking: Enable thinking/reasoning mode

    Returns:
        Complete response text, or None on error
    """
    if _is_anthropic():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return _call_anthropic_stream(messages, timeout_sec, on_token=on_token, thinking=thinking)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": 4096},
        "think": thinking,
    }

    # Use ``with`` so the streaming response (and the underlying TCP
    # connection) is released even if iter_lines exits early via an
    # exception or the caller stops consuming. Without this an aborted
    # stream pinned the connection until GC, which could happen many
    # turns later under sustained reply load.
    try:
        with requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_sec,
            stream=True,
        ) as resp:
            resp.raise_for_status()

            full_response = []
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and isinstance(data["message"], dict):
                            content = data["message"].get("content", "")
                            if content:
                                full_response.append(content)
                                if on_token:
                                    on_token(content)
                    except json.JSONDecodeError:
                        continue

            result = "".join(full_response)
            return result if result.strip() else None

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract text from LLM response - supports multiple response formats."""
    # Preferred: Ollama chat non-stream format
    if "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content

    # Fallback: OpenAI-style format
    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
            elif "text" in choice:
                content = choice["text"]
                if isinstance(content, str):
                    return content

    # Another fallback: direct "content" field
    if "content" in data:
        content = data["content"]
        if isinstance(content, str):
            return content

    return None


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Send an arbitrary messages array to the LLM and return the raw response JSON.
    Caller is responsible for interpreting assistant content (including JSON/tool calls).

    Args:
        base_url: Ollama base URL (ignored when provider is "anthropic")
        chat_model: Model name (ignored when provider is "anthropic")
        messages: Conversation messages
        timeout_sec: Request timeout
        extra_options: Additional Ollama model options (ignored on Anthropic)
        tools: Optional list of tools in OpenAI-compatible JSON schema format.
            On Anthropic these are translated to ``input_schema`` form.
        thinking: Enable thinking/reasoning mode

    Returns the parsed JSON response dict on success, or None on error/timeout.
    Anthropic responses are normalised to the Ollama chat shape
    (``{message: {content, tool_calls}}``) so callers do not need to branch.
    """
    if _is_anthropic():
        return _call_anthropic_chat(messages, timeout_sec, tools=tools, thinking=thinking)

    # Main agentic chat uses 8192 so the system prompt (tool list + protocol
    # guidance + memory context) doesn't overflow and force ollama to truncate
    # — which previously dropped the tool schema on smaller models like
    # gemma4:e2b, tipping them into their pre-trained tool_code scaffolding.
    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192},
        "think": thinking,
    }
    if extra_options and isinstance(extra_options, dict):
        # Merge shallowly into options
        payload["options"].update(extra_options)

    # Add tools for native tool calling support (Ollama 0.4+)
    if tools and isinstance(tools, list) and len(tools) > 0:
        payload["tools"] = tools

    try:
        with requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec) as resp:
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict):
            return data
    except requests.exceptions.Timeout:
        print("  ⏱️ LLM request timed out", flush=True)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ LLM connection error: {e}", flush=True)
        return None
    except requests.exceptions.HTTPError as e:
        # Raise a specific error when the model rejects the tools parameter (HTTP 400).
        # This lets the caller fall back to text-based tool calling automatically.
        if e.response is not None and e.response.status_code == 400 and tools:
            raise ToolsNotSupportedError(
                f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
            )
        print(f"  ❌ LLM HTTP error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"  ❌ LLM error: {e}", flush=True)
        return None

    return None
