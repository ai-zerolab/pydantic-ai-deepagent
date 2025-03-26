from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from itertools import chain
from typing import Literal, assert_never, cast, overload

import httpx
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
from openai.types import ChatModel, chat
from openai.types.chat import ChatCompletionChunk
from pydantic_ai import UnexpectedModelBehavior, result, usage
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

from pydantic_ai_deepagent.reasoning import _utils


def _map_usage(response: chat.ChatCompletion | ChatCompletionChunk) -> usage.Usage:
    response_usage = response.usage
    if response_usage is None:
        return usage.Usage()
    else:
        details: dict[str, int] = {}
        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))
        if response_usage.prompt_tokens_details is not None:
            details.update(response_usage.prompt_tokens_details.model_dump(exclude_none=True))
        return usage.Usage(
            request_tokens=response_usage.prompt_tokens,
            response_tokens=response_usage.completion_tokens,
            total_tokens=response_usage.total_tokens,
            details=details,
        )


@cache
def cached_async_http_client(timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    """Cached HTTPX async client so multiple agents and calls can share the same client.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to, this allows multiple Agents to use a single client.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={"User-Agent": get_user_agent()},
    )


@cache
def get_user_agent() -> str:
    """Get the user agent string for the HTTP client."""
    from .. import __version__

    return f"pydantic-ai-deepagent/{__version__}"


class DeepseekModelSettings(ModelSettings):
    """Settings used for an Deepseek R1 model request."""

    openai_reasoning_effort: chat.ChatCompletionReasoningEffort
    """
    Comming soon
    """


OpenAISystemPromptRole = Literal["system", "developer", "user"]


class DeepseekReasoningModel(Model):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-reasoner",
        *,
        system_prompt: str | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
    ):
        self._model_name = model_name
        self.api_key = api_key
        self._base_url = base_url
        self.system_prompt = system_prompt or self.get_default_system_prompt()

        self.client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=api_key,
            http_client=cached_async_http_client(),  # Use the private attribute
        )
        self.system_prompt_role = system_prompt_role

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
    ) -> tuple[ModelResponse, usage.Usage]:
        """Make a request to the model."""
        response = await self._completions_create(
            messages,
            False,
            cast(DeepseekModelSettings, model_settings or {}),
            model_request_parameters,
        )
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        response = await self._completions_create(
            messages,
            True,
            cast(DeepseekModelSettings, model_settings or {}),
            model_request_parameters,
        )
        async with response:
            yield await self._process_streamed_response(response)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: DeepseekModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]:
        pass

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: DeepseekModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion:
        pass

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: DeepseekModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        openai_messages = list(chain(*(self._map_message(m) for m in messages)))

        return await self.client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            n=1,
            stream=stream,
            stream_options={"include_usage": True} if stream else NOT_GIVEN,
            max_tokens=model_settings.get("max_tokens", NOT_GIVEN),
            temperature=model_settings.get("temperature", NOT_GIVEN),
            top_p=model_settings.get("top_p", NOT_GIVEN),
            timeout=model_settings.get("timeout", NOT_GIVEN),
            seed=model_settings.get("seed", NOT_GIVEN),
            presence_penalty=model_settings.get("presence_penalty", NOT_GIVEN),
            frequency_penalty=model_settings.get("frequency_penalty", NOT_GIVEN),
            logit_bias=model_settings.get("logit_bias", NOT_GIVEN),
            reasoning_effort=model_settings.get("openai_reasoning_effort", NOT_GIVEN),
        )

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        if choice.message.content is not None:
            if (
                hasattr(choice.message, "reasoning_content")
                and choice.message.reasoning_content is not None
            ):
                items.append(
                    TextPart(
                        f"<Thinking>{choice.message.reasoning_content}<\Thinking>\n\n{choice.message.content}"
                    )
                )
            else:
                items.append(TextPart(choice.message.content))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(c.function.name, c.function.arguments, c.id))
        return ModelResponse(items, model_name=response.model, timestamp=timestamp)

    async def _process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk]
    ) -> OpenAIStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior("Streamed response ended without content or tool calls")

        return OpenAIStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
        )

    def _map_message(self, message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        if isinstance(message, ModelRequest):
            yield from self._map_user_message(message)
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(self._map_tool_call(item))
                else:
                    assert_never(item)
            message_param = chat.ChatCompletionAssistantMessageParam(role="assistant")
            if texts:
                # Note: model responses from this model should only have one text item, so the following
                # shouldn't merge multiple texts into one unless you switch models between runs:
                message_param["content"] = "\n\n".join(texts)
            if tool_calls:
                message_param["tool_calls"] = tool_calls
            yield message_param
        else:
            assert_never(message)

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
        return chat.ChatCompletionMessageToolCallParam(
            id=_utils.guard_tool_call_id(t=t, model_source="OpenAI"),
            type="function",
            function={"name": t.tool_name, "arguments": t.args_as_json_str()},
        )

    def _map_user_message(self, message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                if self.system_prompt_role == "developer":
                    yield chat.ChatCompletionDeveloperMessageParam(
                        role="developer", content=self.system_prompt
                    )
                elif self.system_prompt_role == "user":
                    yield chat.ChatCompletionUserMessageParam(
                        role="user", content=self.system_prompt
                    )
                else:
                    yield chat.ChatCompletionSystemMessageParam(
                        role="system", content=self.system_prompt
                    )
            elif isinstance(part, UserPromptPart):
                yield chat.ChatCompletionUserMessageParam(role="user", content=part.content)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=_utils.guard_tool_call_id(t=part, model_source="OpenAI"),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(
                        role="user", content=part.model_response()
                    )
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=_utils.guard_tool_call_id(t=part, model_source="OpenAI"),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)


@dataclass
class OpenAIStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI models."""

    _model_name: str
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += _map_usage(chunk)

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            # Handle the text part of the response
            content = choice.delta.content
            if content is not None:
                yield self._parts_manager.handle_text_delta(
                    vendor_part_id="content", content=content
                )

            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event
