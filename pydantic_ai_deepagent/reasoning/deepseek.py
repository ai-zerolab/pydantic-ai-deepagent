from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic_ai import result
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings


class DeepseekReasoningModel(Model):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        *,
        system_prompt: str | None = None,
    ):
        self._model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt or self.get_default_system_prompt()

    @staticmethod
    def get_default_system_prompt() -> str:
        """
        Deepseek R1 recommends empty system prompt
        """
        return ""

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider, ex: openai."""
        return "Deepseek"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, result.Usage]:
        """Make a request to the model."""
        raise NotImplementedError()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a request to the model and return a streaming response."""
        # This method is not required, but you need to implement it if you want to support streamed responses
        raise NotImplementedError(
            f"Streamed requests not supported by this {self.__class__.__name__}"
        )
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover
