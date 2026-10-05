import os

import pytest

from sizeagent.agent import providers
from sizeagent.agent.providers import ChatClient, FallbackClient, load_dotenv


def _load_scan():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("scan", Path(__file__).parent.parent / "scripts" / "scan_secrets.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_load_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nSA_TEST_A=from_file\nSA_TEST_B='quoted'\n")
    monkeypatch.setenv("SA_TEST_A", "from_env")
    monkeypatch.delenv("SA_TEST_B", raising=False)
    load_dotenv(f)
    assert os.environ["SA_TEST_A"] == "from_env" and os.environ["SA_TEST_B"] == "quoted"
    monkeypatch.delenv("SA_TEST_B")


def test_missing_key_message_names_the_variable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(providers, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        ChatClient("groq")


def test_fallback_switches_on_failure_and_stops_at_last(monkeypatch, capsys):
    calls = []

    class Fake:
        def __init__(self, provider, model=None):
            self.provider, self.model = provider, model or "m"

        def chat(self, messages, tools):
            calls.append(self.provider)
            if self.provider != "ollama":
                raise RuntimeError("HTTP 429 quota")
            return {"content": "ok"}

    monkeypatch.setattr(providers, "ChatClient", Fake)
    c = FallbackClient(["groq", "gemini", "ollama"])
    assert c.chat([], []) == {"content": "ok"} and calls == ["groq", "gemini", "ollama"]
    assert c.model == "ollama/m"
    c2 = FallbackClient(["groq"])
    with pytest.raises(RuntimeError):
        c2.chat([], [])


def test_no_provider_available_is_actionable(monkeypatch):
    monkeypatch.setattr(providers, "available_providers", lambda: [])
    with pytest.raises(RuntimeError, match="Ollama"):
        FallbackClient()


def test_secret_scanner_flags_keys_but_not_prose():
    scan = _load_scan()
    fake_groq = "gsk_" + "x" * 24
    fake_google = "AQ." + "y" * 40
    assert scan.scan_text(f"key = {fake_groq}") == ["Groq key"]
    assert "Google AI Studio key" in scan.scan_text(fake_google)
    assert scan.scan_text("set GROQ_API_KEY=... and see gsk_ prefix docs") == []
