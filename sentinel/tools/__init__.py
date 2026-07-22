"""Tool registry.

Different agents get different tool sets — the investigator is read-only by
construction, which is a stronger guarantee than asking it nicely not to
break production.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from sentinel.tools.observability import READ_TOOLS
from sentinel.tools.remediation import WRITE_TOOLS

ALL_TOOLS: list[BaseTool] = [*READ_TOOLS, *WRITE_TOOLS]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


def get_tools(*, read_only: bool = False) -> list[BaseTool]:
    return list(READ_TOOLS) if read_only else list(ALL_TOOLS)


def get_tool(name: str) -> BaseTool:
    if name not in TOOLS_BY_NAME:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(TOOLS_BY_NAME)}")
    return TOOLS_BY_NAME[name]


__all__ = ["ALL_TOOLS", "READ_TOOLS", "TOOLS_BY_NAME", "WRITE_TOOLS", "get_tool", "get_tools"]
