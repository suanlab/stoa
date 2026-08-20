"""LLM client: env-var key handling (no network calls)."""
import pytest

from stoa.llm import OpenAIClient


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_reads_key_from_env_not_argument(monkeypatch):
    """The key must come from the environment — never passed/stored as a literal."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = OpenAIClient(model="gpt-4o-mini")
    assert client.model == "gpt-4o-mini"
    assert callable(client.answer_fn())
