from __future__ import annotations

from typing import Any


class OpenAIEmbeddingModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.client = client or self._build_client(api_key=api_key, base_url=base_url)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        data = sorted(response.data, key=lambda item: self._get(item, "index", 0))
        return [list(self._get(item, "embedding", [])) for item in data]

    @staticmethod
    def _build_client(*, api_key: str | None, base_url: str | None) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAIEmbeddingModel.") from exc

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    @staticmethod
    def _get(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)


class OpenAILLMClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.client = client or self._build_client(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        return str(response.choices[0].message.content or "")

    @staticmethod
    def _build_client(*, api_key: str | None, base_url: str | None) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAILLMClient.") from exc

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
