"""LLM backends with a disk cache.

Two backends:
  ClaudeCLIBackend  — drives the locally-authenticated `claude -p` headless CLI
  AnthropicBackend  — official SDK (requires ANTHROPIC_API_KEY / `ant auth login`)

Every query returns a parsed JSON object (the prompts instruct the model to
answer with a single JSON object). Responses are cached on disk keyed by
(model, prompt) so re-runs are free.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import time
import urllib.request
import urllib.error

HERE = pathlib.Path(__file__).resolve().parent
CACHE_DIR = HERE.parent / "results" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

JSON_RE = re.compile(r"\{.*\}", re.S)


class UsageLimitError(RuntimeError):
    """The backend account is out of quota; retrying is pointless."""


def extract_json(text: str) -> dict | None:
    """Pull the last parseable JSON object out of a completion."""
    # fenced block first
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = fences + ([m.group(0)] if (m := JSON_RE.search(text)) else [])
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # try trimming to outermost braces greedily from the end
            continue
    return None


class LLMBackend:
    name = "base"

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def query_json(self, prompt: str, retries: int = 2) -> dict | None:
        key = hashlib.sha1(f"{self.name}\x00{prompt}".encode()).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["parsed"]
        for attempt in range(retries + 1):
            try:
                text = self.complete(prompt)
            except UsageLimitError:
                raise
            except Exception:
                time.sleep(3 * (attempt + 1))
                continue
            parsed = extract_json(text)
            if parsed is not None:
                cache_file.write_text(json.dumps({"raw": text[:4000], "parsed": parsed}))
                return parsed
        # never cache failures — a re-run should retry them
        return None


class ClaudeCLIBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.model = model
        self.name = f"claude-cli/{model}"

    def complete(self, prompt: str) -> str:
        r = subprocess.run(
            ["claude", "-p", "--model", self.model,
             "--disallowed-tools", "*", "--max-turns", "1"],
            input=prompt, capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            blob = (r.stdout + r.stderr).lower()
            if "limit" in blob or "quota" in blob or "rate" in blob:
                raise UsageLimitError(f"claude CLI quota: {(r.stdout + r.stderr)[:300]}")
            raise RuntimeError(
                f"claude CLI failed (rc={r.returncode}): "
                f"stderr={r.stderr[:300]!r} stdout={r.stdout[:300]!r}")
        return r.stdout


class CodexCLIBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.model = model
        self.name = f"codex/{model or 'default'}"

    def complete(self, prompt: str) -> str:
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check"]
        if self.model:
            cmd += ["--model", self.model]
        r = subprocess.run(
            cmd,
            input=prompt, capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            blob = (r.stdout + r.stderr).lower()
            if "limit" in blob or "quota" in blob or "rate" in blob or "insufficient_quota" in blob:
                raise UsageLimitError(f"codex CLI quota: {(r.stdout + r.stderr)[:300]}")
            raise RuntimeError(
                f"codex CLI failed (rc={r.returncode}): "
                f"stderr={r.stderr[:300]!r} stdout={r.stdout[:300]!r}")
        return r.stdout


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"api/{model}"

    def complete(self, prompt: str) -> str:
        kwargs = {}
        if "haiku" not in self.model:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("refusal")
        return "".join(b.text for b in resp.content if b.type == "text")


class OllamaBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        import os

        self.model = model
        self.name = f"ollama/{model}"
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.url = f"{host.rstrip('/')}/api/generate"

    def complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # reasoning models otherwise burn the whole num_predict budget
            # thinking and return an empty response
            "think": False,
            "options": {
                # L3 prompts run ~10k chars; the ollama default ctx (4096)
                # would silently truncate them
                "num_ctx": 16384,
                "num_predict": 8192,
                "temperature": 0,
            },
        }
        for attempt in (0, 1):
            req = urllib.request.Request(
                self.url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return result.get("response", "")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8")[:300]
                if attempt == 0 and e.code == 400 and "think" in detail:
                    body.pop("think", None)  # model rejects the toggle
                    continue
                raise RuntimeError(f"ollama failed (HTTP {e.code}): {detail}")
            except Exception as e:
                raise RuntimeError(f"ollama request failed: {e}")


class OpenAIBackend(LLMBackend):
    """Any OpenAI-compatible /v1/chat/completions server (e.g. vLLM serve).

    spec: openai:<model>@<base_url>, e.g.
          openai:ft-2b@http://localhost:8000/v1
    """

    def __init__(self, model_at_url: str) -> None:
        self.model, _, base = model_at_url.partition("@")
        base = base or "http://localhost:8000/v1"
        self.url = f"{base.rstrip('/')}/chat/completions"
        self.name = f"openai/{self.model}"

    def complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"openai backend failed (HTTP {e.code}): "
                f"{e.read().decode('utf-8')[:300]}")


def make_backend(spec: str) -> LLMBackend:
    """spec: 'cli:haiku' | 'cli:claude-sonnet-5' | 'api:claude-opus-4-8' | 'codex:gpt-4o' | 'ollama:gemma4:31b'."""
    kind, _, model = spec.partition(":")
    if kind == "cli":
        return ClaudeCLIBackend(model)
    if kind == "api":
        return AnthropicBackend(model)
    if kind == "codex":
        return CodexCLIBackend(model)
    if kind == "ollama":
        return OllamaBackend(model)
    if kind == "openai":
        return OpenAIBackend(model)
    raise ValueError(spec)
