"""Minimal OpenAI-compatible chat client for LLM-backed evaluation (docs/research_plan.md §5).

Used to measure the task-utility term U of the reward (environment.reward) — until
now hardcoded to 1.0 — with a real LLM answering under a memory budget. NO GPU: the
LLM is called over an HTTP API, not self-hosted.

SECURITY: the API key is read from the environment ($OPENAI_API_KEY), never taken as
a literal in code or committed. Rotate any key that has been exposed. This module is
NOT imported by stoa/__init__, so `import stoa` needs neither httpx nor a key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_DEFAULT_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


@dataclass
class OpenAIClient:
    """Thin chat-completions wrapper. Requires $OPENAI_API_KEY in the environment."""
    model: str = _DEFAULT_MODEL
    base_url: str = _DEFAULT_BASE
    timeout: float = 30.0
    max_tokens: int = 32
    temperature: float = 0.0
    # Long experiments make thousands of sequential calls, so a single transient network
    # failure ends the run hours in. The representation-axis experiment died at dialogue 3
    # of 5 on one httpx.ReadTimeout with no retry in place.
    max_retries: int = 5
    backoff_s: float = 2.0

    def __post_init__(self) -> None:
        self._key = os.getenv("OPENAI_API_KEY")
        if not self._key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell (do not paste keys "
                "into chat or code): `export OPENAI_API_KEY=sk-...`"
            )

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the assistant's reply text for a single-turn prompt.

        Retries transient failures (timeouts, connection resets, 429 and 5xx) with
        exponential backoff. A 4xx other than 429 is a request we built wrong and is raised
        immediately -- retrying it would just burn quota and hide the bug.
        """
        import time

        import httpx

        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=self.timeout,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=resp.request,
                                                response=resp)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                if e.response is not None and 400 <= e.response.status_code < 500 \
                        and e.response.status_code != 429:
                    raise
                last = e
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
            if attempt + 1 < self.max_retries:
                time.sleep(self.backoff_s * (2 ** attempt))
        raise RuntimeError(f"chat completion failed after {self.max_retries} attempts") from last

    def answer_fn(self):
        """Adapt to the `answer_fn(prompt) -> str` interface used by eval harnesses."""
        return lambda prompt: self.complete(prompt)

    def embed(self, texts: list[str], model: str = "text-embedding-3-small",
              batch: int = 512) -> list[list[float]]:
        """Embed `texts` (batched). Used for the retrieval baselines in eval/retrieval.py.

        Retries transient failures like `complete` does: a long run makes many batch calls
        and one timeout should not end it.
        """
        import time

        import httpx

        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            last: Exception | None = None
            data = None
            for attempt in range(self.max_retries):
                try:
                    resp = httpx.post(
                        f"{self.base_url}/embeddings",
                        headers={"Authorization": f"Bearer {self._key}"},
                        json={"model": model, "input": chunk},
                        timeout=120.0,
                    )
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise httpx.HTTPStatusError("retryable", request=resp.request,
                                                    response=resp)
                    resp.raise_for_status()
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    break
                except httpx.HTTPStatusError as e:
                    if e.response is not None and 400 <= e.response.status_code < 500 \
                            and e.response.status_code != 429:
                        raise
                    last = e
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last = e
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
            if data is None:
                raise RuntimeError(
                    f"embedding failed after {self.max_retries} attempts") from last
            out.extend(d["embedding"] for d in data)
        return out

    def judge_fn(self):
        """Return an LLM judge `(question, gold, reply) -> bool` — a more faithful
        scorer than substring match (handles paraphrase/format, e.g. dates)."""
        def judge(question: str, gold: str, reply: str) -> bool:
            verdict = self.complete(
                f"Question: {question}\nReference answer: {gold}\nCandidate answer: {reply}\n\n"
                "Does the candidate answer convey the same information as the reference? "
                "Answer strictly 'yes' or 'no'.",
                system="You are a strict grader for question answering.",
            )
            return verdict.strip().lower().startswith("y")
        return judge
