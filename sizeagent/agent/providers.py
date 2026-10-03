"""Minimal OpenAI-compatible chat client with tool calling.

Gemini, Groq and Ollama all expose an OpenAI-compatible /chat/completions
endpoint, so one client covers all three free options. No SDK dependency.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = {
    # name: (base_url, api-key env var or None, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.8-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b"),
}


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env reader (no dependency). Real environment variables always win."""
    for f in [Path(path)] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def ollama_up(base: str = "http://localhost:11434") -> bool:
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def available_providers() -> list[str]:
    """Providers usable right now, hosted ones first, local Ollama as the last resort."""
    load_dotenv()
    out = [n for n, (_, env, _) in PROVIDERS.items() if env and os.environ.get(env)]
    return out + (["ollama"] if ollama_up() else [])


@dataclass
class ChatClient:
    provider: str = "gemini"
    model: str | None = None
    temperature: float = 0.2
    max_retries: int = 6

    def __post_init__(self):
        load_dotenv()
        base, key_env, default_model = PROVIDERS[self.provider]
        self.base = os.environ.get("SIZEAGENT_BASE_URL", base)
        self.model = self.model or default_model
        self.key = os.environ.get(key_env, "") if key_env else "ollama"
        if key_env and not self.key:
            raise RuntimeError(f"set {key_env} to use provider '{self.provider}'")

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({"model": self.model, "messages": messages, "tools": tools,
                           "tool_choice": "auto", "temperature": self.temperature}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "sizeagent/0.1 (+https://github.com/Nags-gk/sizeagent)",
                                              "Authorization": f"Bearer {self.key}"})
        delay = 5.0
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                # Free tiers rate-limit aggressively; back off on 429/5xx.
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"{self.provider} HTTP {e.code}: {e.read()[:500]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                raise RuntimeError(f"{self.provider} unreachable: {e}") from e
        raise RuntimeError("unreachable")


class FallbackClient:
    """Tries providers in order and moves to the next one on quota, auth or network
    failures, so a run is not lost when a free tier runs dry."""

    def __init__(self, names: list[str] | None = None, model: str | None = None):
        self.names = names or available_providers()
        if not self.names:
            raise RuntimeError("no provider available: set GROQ_API_KEY / GEMINI_API_KEY in .env "
                               "or start Ollama (`ollama serve`)")
        self.model_override, self.idx = model, 0
        self.client = ChatClient(self.names[0], model)

    @property
    def model(self) -> str | None:
        return f"{self.client.provider}/{self.client.model}"

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        while True:
            try:
                return self.client.chat(messages, tools)
            except RuntimeError as e:
                if self.idx + 1 >= len(self.names):
                    raise
                print(f"[auto] {self.client.provider} failed ({str(e)[:90]}); switching to {self.names[self.idx + 1]}")
                self.idx += 1
                self.client = ChatClient(self.names[self.idx], None)
