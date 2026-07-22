"""A LangChain chat model for OpenRouter, implemented directly on httpx.

Why not langchain-openai? It depends on the `openai` package, which depends on
`jiter`, whose compiled extension is blocked by Windows Smart App Control on
this machine. Rather than weaken an OS security control, we speak the
chat-completions protocol ourselves. It is a well-specified HTTP API and the
implementation is small.

Two things this buys beyond dodging the dependency:
  * a model fallback chain, which matters because OpenRouter's free tier
    returns 429/503 constantly, and
  * full visibility into token usage per call, which feeds the cost dashboard.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from sentinel.logging_setup import get_logger

log = get_logger(__name__)

# OpenRouter free-tier models are frequently saturated. These are transient
# and worth failing over on; anything else is a real error we should surface.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, model: str | None = None):
        super().__init__(message)
        self.status = status
        self.model = model

    @property
    def retryable(self) -> bool:
        return self.status in _RETRYABLE_STATUS if self.status else False


# --- message conversion ---------------------------------------------------


def _lc_to_wire(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": str(m.content),
                }
            )
        elif isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in m.tool_calls
                ]
                # Some providers reject a null content alongside tool_calls.
                msg["content"] = m.content or ""
            out.append(msg)
        else:
            out.append({"role": "user", "content": str(m.content)})
    return out


def _parse_tool_calls(raw: list[dict] | None) -> list[dict[str, Any]]:
    calls = []
    for i, tc in enumerate(raw or []):
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            # A model emitting malformed JSON is common enough to handle rather
            # than crash on. Surface it as an arg the tool layer will reject.
            log.warning("openrouter.bad_tool_args", raw=raw_args[:200])
            args = {"__malformed__": raw_args}
        calls.append(
            {
                "name": fn.get("name", ""),
                "args": args if isinstance(args, dict) else {"value": args},
                "id": tc.get("id") or f"call_{i}",
                "type": "tool_call",
            }
        )
    return calls


def _wire_to_ai_message(data: dict[str, Any], model: str) -> AIMessage:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    usage = data.get("usage") or {}
    return AIMessage(
        content=msg.get("content") or "",
        tool_calls=_parse_tool_calls(msg.get("tool_calls")),
        response_metadata={
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason"),
        },
        usage_metadata={
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )


# --- the model ------------------------------------------------------------


class ChatOpenRouter(BaseChatModel):
    """Chat model backed by OpenRouter, with a model fallback chain."""

    model: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 180.0
    # Tried in order when the primary returns a retryable error.
    fallback_models: list[str] = Field(default_factory=list)
    referer: str = "https://github.com/GautamRaju18/Sentinel"
    title: str = "Sentinel"

    _bound_tools: list[dict] | None = PrivateAttr(default=None)
    _tool_choice: Any = PrivateAttr(default=None)
    _response_format: dict | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    # --- configuration ----------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseTool],
        *,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        clone = self._clone()
        clone._bound_tools = [convert_to_openai_tool(t) for t in tools]
        clone._tool_choice = tool_choice
        return clone

    def with_structured_output(
        self,
        schema: type,
        *,
        method: Literal["json_schema", "json_mode"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Constrain output to a Pydantic schema.

        Prefers the provider's json_schema mode; falls back to json_mode plus a
        parser, since free-tier providers vary in what they honour.
        """
        from langchain_core.output_parsers import PydanticOutputParser

        clone = self._clone()
        name = getattr(schema, "__name__", "Response")
        if method == "json_schema":
            clone._response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }
        else:
            clone._response_format = {"type": "json_object"}
        parser = PydanticOutputParser(pydantic_object=schema)
        if include_raw:
            return clone | {"raw": lambda m: m, "parsed": parser}
        return clone | parser

    def _clone(self) -> ChatOpenRouter:
        clone = self.__class__(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            fallback_models=list(self.fallback_models),
        )
        clone._bound_tools = self._bound_tools
        clone._tool_choice = self._tool_choice
        clone._response_format = self._response_format
        return clone

    # --- request building -------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

    def _payload(
        self, messages: Sequence[BaseMessage], model: str, stop: list[str] | None, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": _lc_to_wire(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if stop:
            body["stop"] = stop
        if stream:
            body["stream"] = True
        if self._bound_tools:
            body["tools"] = self._bound_tools
            if self._tool_choice:
                body["tool_choice"] = self._tool_choice
        if self._response_format:
            body["response_format"] = self._response_format
        return body

    def _candidates(self) -> list[str]:
        return [self.model, *self.fallback_models]

    @staticmethod
    def _raise_for_body(data: dict, model: str) -> None:
        if "error" in data:
            err = data["error"]
            raise OpenRouterError(str(err.get("message", err)), status=err.get("code"), model=model)

    # --- sync -------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last: Exception | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for model in self._candidates():
                try:
                    r = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=self._payload(messages, model, stop, stream=False),
                    )
                    data = r.json()
                    self._raise_for_body(data, model)
                    r.raise_for_status()
                    return ChatResult(
                        generations=[ChatGeneration(message=_wire_to_ai_message(data, model))]
                    )
                except (OpenRouterError, httpx.HTTPError) as e:
                    last = e
                    status = getattr(e, "status", None) or getattr(
                        getattr(e, "response", None), "status_code", None
                    )
                    if status not in _RETRYABLE_STATUS:
                        raise
                    log.warning("openrouter.failover", model=model, status=status)
        raise OpenRouterError(
            f"all models exhausted ({', '.join(self._candidates())}): {last}"
        ) from last

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, self.model, stop, stream=True),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text)
                    yield chunk

    # --- async ------------------------------------------------------------

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for model in self._candidates():
                try:
                    r = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=self._payload(messages, model, stop, stream=False),
                    )
                    data = r.json()
                    self._raise_for_body(data, model)
                    r.raise_for_status()
                    return ChatResult(
                        generations=[ChatGeneration(message=_wire_to_ai_message(data, model))]
                    )
                except (OpenRouterError, httpx.HTTPError) as e:
                    last = e
                    status = getattr(e, "status", None) or getattr(
                        getattr(e, "response", None), "status_code", None
                    )
                    if status not in _RETRYABLE_STATUS:
                        raise
                    log.warning("openrouter.failover", model=model, status=status)
        raise OpenRouterError(
            f"all models exhausted ({', '.join(self._candidates())}): {last}"
        ) from last

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, self.model, stop, stream=True),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text)
                    yield chunk


def _parse_sse_line(line: str) -> ChatGenerationChunk | None:
    """Turn one SSE line into a chunk, or None if it carries no content."""
    if not line or not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content") or ""
    if not content:
        return None
    return ChatGenerationChunk(message=AIMessageChunk(content=content))
