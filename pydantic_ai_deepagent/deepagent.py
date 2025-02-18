from collections.abc import AsyncIterator

from pydantic_ai import result, usage
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


class DeepAgentModel(Model):
    def __init__(
        self,
        reasoning_model: Model,
        execution_model: Model,
    ):
        self.reasoning_model = reasoning_model
        self.execution_model = execution_model

    @property
    def model_name(self) -> str:
        """The model name."""
        return f"{self.reasoning_model.model_name}-{self.execution_model.model_name}"

    @property
    def system(self) -> str | None:
        """The system / model provider, ex: openai."""
        raise f"{self.reasoning_model.system}-{self.execution_model.system}"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        """Make a request to the model."""
        reasoning_response, reasoning_usage = await self.reasoning_model.request(
            messages, model_settings, model_request_parameters
        )

        messages.append(reasoning_response)
        messages.append(
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="Please use a tool accroding the reasoning result.",
                    )
                ]
            )
        )

        execution_response, execution_usage = await self.execution_model.request(
            messages, model_settings, model_request_parameters
        )

        total_usage = reasoning_usage + execution_usage
        total_usage.details = {
            **(total_usage.details or {}),
            "reasoning_request_tokens": reasoning_usage.request_tokens,
            "reasoning_response_tokens": reasoning_usage.response_tokens,
            "reasoning_total_tokens": reasoning_usage.total_tokens,
            "execution_request_tokens": execution_usage.request_tokens,
            "execution_response_tokens": execution_usage.response_tokens,
            "execution_total_tokens": execution_usage.total_tokens,
        }

        return (execution_response, total_usage)

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
