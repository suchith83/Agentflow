import os

from .base_embedding import BaseEmbedding


HAS_OPENAI = False

try:
    from openai import AsyncOpenAI, OpenAIError

    HAS_OPENAI = True
except ImportError as e:
    raise ImportError(
        "The 'openai' package is required for OpenAIEmbedding. "
        "Please install it via 'pip install openai'."
    ) from e


class OpenAIEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        OPENAI_API_KEY: str | None = None,
    ) -> None:
        if not HAS_OPENAI:
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbedding. "
                "Please install it via 'pip install openai'."
            )
        self.model = model
        if OPENAI_API_KEY:
            self.api_key = OPENAI_API_KEY
        elif "OPENAI_API_KEY" in os.environ:
            self.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError(
                "OpenAI API key must be provided via parameter or OPENAI_API_KEY env var"
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
        )

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model,
            )
            return [data.embedding for data in response.data]
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    async def aembed(self, text: str) -> list[float]:
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
            )
            return response.data[0].embedding if response.data else []
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    @property
    def dimension(self) -> int:
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 1536,
            "text-embedding-3-xl": 1536,
            "text-embedding-4-base": 8192,
            "text-embedding-4-large": 8192,
        }
        if self.model in model_dimensions:
            return model_dimensions[self.model]
        raise ValueError(f"Unknown model '{self.model}'. Cannot determine dimension.")
