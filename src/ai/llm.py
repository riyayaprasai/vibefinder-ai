"""Optional local LLM client for Ollama (https://ollama.com), using only the standard library.

Nothing here is required: the app's default backend is "offline". Ollama is free and runs on
your own machine, so no paid API or API key is involved. Configure with environment variables:
    OLLAMA_HOST             default http://localhost:11434
    VIBEFINDER_OLLAMA_MODEL default llama3.2:3b
"""

import json
import os
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


class LLMUnavailable(RuntimeError):
    pass


class OllamaClient:
    name = "ollama"

    def __init__(self, model: str = None, host: str = None, timeout: float = 120.0):
        self.model = model or os.environ.get("VIBEFINDER_OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        """True when the Ollama server answers and the configured model is pulled."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=1.5) as resp:
                models = json.loads(resp.read().decode("utf-8")).get("models", [])
        except (urllib.error.URLError, OSError, ValueError):
            return False
        names = {m.get("name", "") for m in models}
        return self.model in names or f"{self.model}:latest" in names

    def complete(self, prompt: str, json_mode: bool = False, temperature: float = 0.0) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": temperature, "seed": 7}}
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(f"{self.host}/api/generate", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8")).get("response", "")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMUnavailable(f"Ollama request failed: {exc}") from exc
