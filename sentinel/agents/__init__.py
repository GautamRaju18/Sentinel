"""A minimal ReAct-style tool-calling loop.

Written out by hand rather than pulled from a prebuilt helper because the loop
IS the concept: bind tools, call the model, execute whatever it asked for, feed
the results back, repeat until it stops asking. Phase 3 replaces this with a
LangGraph StateGraph that adds branching, cycles, checkpointing and approval
gates — but this is the kernel underneath all of that.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from sentinel.config import get_settings
from sentinel.logging_setup import get_logger
from sentinel.models.router import ModelTier, get_model

log = get_logger(__name__)


@dataclass
class ToolInvocation:
    name: str
    args: dict[str, Any]
    result: str
    duration_ms: float
    error: str | None = None


@dataclass
class AgentRun:
    messages: list[BaseMessage] = field(default_factory=list)
    invocations: list[ToolInvocation] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_because: str = ""

    @property
    def final_text(self) -> str:
        for m in reversed(self.messages):
            if isinstance(m, AIMessage) and m.content:
                return str(m.content)
        return ""

    @property
    def tool_sequence(self) -> list[str]:
        """The trajectory — what evals in Phase 7 score against."""
        return [i.name for i in self.invocations]


async def _run_tool(tool: BaseTool, call: dict[str, Any]) -> ToolInvocation:
    started = time.perf_counter()
    name, args = call["name"], call.get("args", {})
    try:
        result = await asyncio.wait_for(
            tool.ainvoke(args), timeout=get_settings().tool_timeout_seconds
        )
        return ToolInvocation(name, args, str(result), (time.perf_counter() - started) * 1000)
    except TimeoutError:
        msg = f"Tool '{name}' timed out after {get_settings().tool_timeout_seconds}s."
        return ToolInvocation(name, args, msg, (time.perf_counter() - started) * 1000, msg)
    except Exception as e:
        # Errors go back to the model as observations rather than crashing the
        # run — a good agent recovers from a bad argument by trying again.
        msg = f"Tool '{name}' failed: {type(e).__name__}: {e}"
        log.warning("tool.error", tool=name, error=str(e))
        return ToolInvocation(name, args, msg, (time.perf_counter() - started) * 1000, msg)


async def run_agent(
    task: str,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    tier: ModelTier = ModelTier.WORKER,
    max_steps: int = 12,
    on_event: Callable[[str, dict], None] | None = None,
) -> AgentRun:
    """Run the loop until the model stops calling tools or we hit max_steps."""
    by_name = {t.name: t for t in tools}
    model = get_model(tier).bind_tools(list(tools))

    run = AgentRun(messages=[SystemMessage(system_prompt), HumanMessage(task)])

    def emit(kind: str, payload: dict) -> None:
        if on_event:
            on_event(kind, payload)

    for step in range(1, max_steps + 1):
        run.steps = step
        emit("step", {"step": step})

        response: AIMessage = await model.ainvoke(run.messages)
        run.messages.append(response)

        if usage := getattr(response, "usage_metadata", None):
            run.input_tokens += usage.get("input_tokens", 0)
            run.output_tokens += usage.get("output_tokens", 0)

        calls = response.tool_calls or []
        if not calls:
            run.stopped_because = "model produced a final answer"
            emit("final", {"text": str(response.content)})
            return run

        emit("tool_calls", {"names": [c["name"] for c in calls]})

        # Parallel tool calls: the model asked for several independent reads,
        # so there is no reason to serialise them.
        results = await asyncio.gather(
            *[
                _run_tool(by_name[c["name"]], c)
                if c["name"] in by_name
                else _unknown_tool(c, sorted(by_name))
                for c in calls
            ]
        )

        for call, inv in zip(calls, results, strict=True):
            run.invocations.append(inv)
            run.messages.append(ToolMessage(content=inv.result, tool_call_id=call["id"]))
            emit(
                "tool_result",
                {"name": inv.name, "ms": round(inv.duration_ms), "preview": inv.result[:200]},
            )

    run.stopped_because = f"hit max_steps={max_steps}"
    log.warning("agent.max_steps", max_steps=max_steps)
    return run


async def _unknown_tool(call: dict, available: list[str]) -> ToolInvocation:
    msg = f"No tool named '{call['name']}'. Available tools: {', '.join(available)}"
    return ToolInvocation(call["name"], call.get("args", {}), msg, 0.0, msg)
