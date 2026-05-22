"""
Tests for setup_wizard.get_required_models()'s provider awareness.

The wizard helper drives which Ollama models the user is asked to pull on
first launch. Anthropic provider mode collapses every chat slot onto a
Claude model in the cloud, so only the embedding model needs to be pulled
locally — the 7-12 GB chat-model download is skipped entirely.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Point JARVIS_CONFIG_PATH at a temp file for each test."""
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(tmp_path / "config.json"))


def _write_config(monkeypatch, **kwargs):
    """Materialise a config.json the wizard helper will read via load_settings()."""
    import os
    cfg_path = os.environ["JARVIS_CONFIG_PATH"]
    with open(cfg_path, "w") as f:
        json.dump(kwargs, f)


class TestGetRequiredModelsProviderAware:
    def test_anthropic_mode_returns_embedding_only(self, monkeypatch):
        _write_config(
            monkeypatch,
            llm_provider="anthropic",
            anthropic_api_key="sk-test",
            ollama_embed_model="nomic-embed-text",
            ollama_chat_model="gpt-oss:20b",
            intent_judge_model="gemma4:e2b",
        )
        # Import inside the test so the config-path env var is already set.
        from desktop_app.setup_wizard import get_required_models

        models = get_required_models()
        # Anthropic mode: no chat model, no intent judge model — those run on Claude.
        assert "gpt-oss:20b" not in models
        assert "gemma4:e2b" not in models
        # Embedding is still required because Anthropic has no embeddings API.
        assert "nomic-embed-text" in models
        # And it's the ONLY required model.
        assert models == ["nomic-embed-text"]

    def test_ollama_mode_returns_chat_embed_and_intent_judge(self, monkeypatch):
        _write_config(
            monkeypatch,
            llm_provider="ollama",
            ollama_embed_model="nomic-embed-text",
            ollama_chat_model="gemma4:e4b",
            intent_judge_model="gemma4:e2b",
        )
        from desktop_app.setup_wizard import get_required_models

        models = get_required_models()
        assert "gemma4:e4b" in models
        assert "nomic-embed-text" in models
        assert "gemma4:e2b" in models

    def test_ollama_mode_dedupes_when_chat_equals_intent_judge(self, monkeypatch):
        _write_config(
            monkeypatch,
            llm_provider="ollama",
            ollama_embed_model="nomic-embed-text",
            ollama_chat_model="gemma4:e2b",
            intent_judge_model="gemma4:e2b",
        )
        from desktop_app.setup_wizard import get_required_models

        models = get_required_models()
        # The chat model and the intent judge happen to share an ID — we should
        # only download it once, not list it twice.
        assert models.count("gemma4:e2b") == 1
        assert "nomic-embed-text" in models

    def test_unknown_provider_defaults_to_ollama_behaviour(self, monkeypatch):
        _write_config(
            monkeypatch,
            llm_provider="cohere",
            ollama_embed_model="nomic-embed-text",
            ollama_chat_model="gemma4:e2b",
            intent_judge_model="gemma4:e2b",
        )
        from desktop_app.setup_wizard import get_required_models

        # load_settings sanitises unknown providers to "ollama", so the chat
        # model is still expected to be pulled.
        models = get_required_models()
        assert "gemma4:e2b" in models
        assert "nomic-embed-text" in models
