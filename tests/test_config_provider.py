"""
Tests for the LLM provider config fields added so jarvis can talk to Anthropic.

Covers defaults, parsing, and the Anthropic supported-model list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import (
    DEFAULT_ANTHROPIC_MODEL,
    SUPPORTED_ANTHROPIC_MODELS,
    get_default_config,
    load_settings,
)


class TestProviderDefaults:
    def test_default_provider_is_ollama(self):
        cfg = get_default_config()
        assert cfg["llm_provider"] == "ollama"

    def test_anthropic_api_key_defaults_to_empty(self):
        cfg = get_default_config()
        assert cfg["anthropic_api_key"] == ""

    def test_default_anthropic_chat_model_is_sonnet(self):
        cfg = get_default_config()
        assert cfg["anthropic_chat_model"] == "claude-sonnet-4-6"
        assert DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-4-6"

    def test_anthropic_max_tokens_default(self):
        cfg = get_default_config()
        assert cfg["anthropic_max_tokens"] >= 1024

    def test_anthropic_base_url_default(self):
        cfg = get_default_config()
        assert cfg["anthropic_base_url"].startswith("https://api.anthropic.com")


class TestSupportedAnthropicModels:
    def test_includes_claude_sonnet_4_6(self):
        assert "claude-sonnet-4-6" in SUPPORTED_ANTHROPIC_MODELS

    def test_includes_claude_opus_4_7(self):
        assert "claude-opus-4-7" in SUPPORTED_ANTHROPIC_MODELS

    def test_includes_claude_haiku_4_5(self):
        # Haiku 4.5 uses a dated alias per Anthropic's naming.
        assert any("haiku-4-5" in m for m in SUPPORTED_ANTHROPIC_MODELS)

    def test_entries_have_required_fields(self):
        required = {"name", "description"}
        for model_id, info in SUPPORTED_ANTHROPIC_MODELS.items():
            assert isinstance(info, dict)
            for field in required:
                assert field in info, f"{model_id} missing {field}"


class TestLoadSettingsProvider:
    def test_load_settings_with_anthropic_provider(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "llm_provider": "anthropic",
            "anthropic_api_key": "sk-mykey",
            "anthropic_chat_model": "claude-opus-4-7",
            "anthropic_max_tokens": 8192,
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_file))

        s = load_settings()
        assert s.llm_provider == "anthropic"
        assert s.anthropic_api_key == "sk-mykey"
        assert s.anthropic_chat_model == "claude-opus-4-7"
        assert s.anthropic_max_tokens == 8192

    def test_unknown_provider_value_falls_back_to_ollama(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"llm_provider": "cohere"}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_file))

        s = load_settings()
        assert s.llm_provider == "ollama"

    def test_anthropic_api_key_strips_whitespace(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "llm_provider": "anthropic",
            "anthropic_api_key": "  sk-foo  ",
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_file))

        s = load_settings()
        assert s.anthropic_api_key == "sk-foo"
