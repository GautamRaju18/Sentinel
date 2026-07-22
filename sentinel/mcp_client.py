"""MCP client — load tools from external MCP servers into the agent.

The other half of Phase 2. The server module publishes Sentinel's own tools;
this module consumes tools published by anyone else, and hands them to
LangGraph as ordinary LangChain tools.

The design point: an agent's capabilities become a deployment concern rather
than a code concern. Adding filesystem access to the code agent is an edit to
mcp.config.json, not a new module.

External servers are configured in mcp.config.json at the repo root. Missing
servers degrade gracefully — a machine without Node should still run the whole
project, just without the Node-based servers.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from sentinel.config import PROJECT_ROOT, get_settings
from sentinel.logging_setup import get_logger

log = get_logger(__name__)

CONFIG_PATH = PROJECT_ROOT / "mcp.config.json"


def _expand(value: Any) -> Any:
    """Resolve ${VAR} against .env so tokens never sit in the config file."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        settings = get_settings()
        return getattr(settings, name.lower(), None) or os.environ.get(name, "")
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config() -> dict[str, dict]:
    if not CONFIG_PATH.exists():
        log.warning("mcp.no_config", path=str(CONFIG_PATH))
        return {}
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {k: _expand(v) for k, v in raw.get("mcpServers", {}).items()}


def _is_available(name: str, spec: dict) -> tuple[bool, str]:
    if spec.get("disabled"):
        return False, "disabled in config"
    command = spec.get("command")
    if not command:
        return True, "http transport"
    if shutil.which(command) is None:
        return False, f"'{command}' not on PATH"
    for var in spec.get("requires_env", []):
        if not getattr(get_settings(), var.lower(), ""):
            return False, f"{var} is not set"
    return True, "ok"


def describe_servers() -> str:
    lines = ["Configured MCP servers:"]
    for name, spec in load_config().items():
        ok, why = _is_available(name, spec)
        mark = "✓" if ok else "·"
        lines.append(f"  {mark} {name:<14} {spec.get('command', spec.get('url', '?'))}  [{why}]")
    return "\n".join(lines) if len(lines) > 1 else "No MCP servers configured."


async def load_mcp_tools(
    servers: list[str] | None = None, *, namespace: bool = True
) -> list[BaseTool]:
    """Connect to configured MCP servers and return their tools.

    Args:
        servers: Server names to load. None loads every available server.
        namespace: Prefix tool names with the server name. Prevents collisions
            when two servers both publish a tool called `search`.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    config = load_config()
    wanted = servers or list(config)
    connections: dict[str, dict] = {}

    for name in wanted:
        spec = config.get(name)
        if spec is None:
            log.warning("mcp.unknown_server", server=name)
            continue
        ok, why = _is_available(name, spec)
        if not ok:
            log.info("mcp.skip", server=name, reason=why)
            continue
        conn = {k: v for k, v in spec.items() if k not in {"disabled", "requires_env"}}
        conn.setdefault("transport", "stdio" if "command" in conn else "streamable_http")
        connections[name] = conn

    if not connections:
        log.warning("mcp.no_servers_available")
        return []

    try:
        client = MultiServerMCPClient(connections)
        tools = await client.get_tools()
    except Exception as e:
        # A broken external server must not take down the incident response
        # system. Degrade to the built-in toolset and say so.
        log.error("mcp.connect_failed", error=str(e), servers=list(connections))
        return []

    if namespace and len(connections) > 1:
        for t in tools:
            server = getattr(t, "_mcp_server", None)
            if server and not t.name.startswith(f"{server}_"):
                t.name = f"{server}_{t.name}"

    log.info("mcp.loaded", servers=list(connections), tool_count=len(tools))
    return tools


async def load_all_tools(*, read_only: bool = True) -> list[BaseTool]:
    """Built-in tools plus everything the configured MCP servers publish."""
    from sentinel.tools import get_tools

    builtin = get_tools(read_only=read_only)
    external = await load_mcp_tools()
    if external:
        log.info("tools.merged", builtin=len(builtin), external=len(external))
    return [*builtin, *external]


def ensure_default_config() -> Path:
    """Write a starter mcp.config.json if none exists."""
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    default = {
        "$comment": (
            "External MCP servers. ${VAR} resolves against .env. Servers whose "
            "command is missing from PATH are skipped with a log line rather "
            "than failing the run."
        ),
        "mcpServers": {
            "sentinel": {
                # sys.executable, not "python" — the bare name resolves to
                # whatever is first on PATH, which is not this venv and will
                # not have the dependencies.
                "command": sys.executable,
                "args": ["-m", "sentinel.mcp_server.server"],
                "cwd": str(PROJECT_ROOT),
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(PROJECT_ROOT)],
            },
            "git": {
                "command": "uvx",
                "args": ["mcp-server-git", "--repository", str(PROJECT_ROOT)],
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
                "requires_env": ["GITHUB_TOKEN"],
                "disabled": True,
                "$comment": "Enable once the PAT has Contents: read/write.",
            },
        },
    }
    CONFIG_PATH.write_text(json.dumps(default, indent=2), encoding="utf-8")
    return CONFIG_PATH
