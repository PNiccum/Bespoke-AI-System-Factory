"""Call the factory's tools through its MCP server instead of in-process.

``mcp_function`` turns a tool agent's entrypoint into a function with the same signature that sends the call to the
server and validates the reply against the entrypoint's return type, so the orchestrator cannot tell the two apart.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import typing
from collections.abc import Callable

from mcp import Client, StdioServerParameters
from pydantic import BaseModel

from factory.tools.sandbox import scrubbed_env


class McpToolError(RuntimeError):
    pass


class McpTools:
    """A connection recipe for the factory's MCP server. Each call starts the server, makes one call, and stops it:
    slower than a long-lived session, but nothing is left running if a run is abandoned."""

    def __init__(self, roots: list[str], server: StdioServerParameters | None = None, timeout_s: float = 3600):
        env = scrubbed_env() | {"FACTORY_MCP_ROOTS": os.pathsep.join(os.path.abspath(r) for r in roots)}
        self.server = server or StdioServerParameters(command=sys.executable, args=["-m", "factory.mcp"],
                                                      env=env, cwd=os.getcwd())
        self.timeout_s = timeout_s

    async def _call(self, name: str, arguments: dict) -> dict:
        async with Client(self.server, read_timeout_seconds=self.timeout_s) as client:
            result = await client.call_tool(name, arguments)
        if result.is_error:
            raise McpToolError(f"{name}: " + " ".join(getattr(c, "text", "") for c in result.content))
        return result.structured_content or {}

    def call(self, name: str, arguments: dict) -> dict:
        return asyncio.run(self._call(name, arguments))


def mcp_function(fn: Callable, tools: McpTools) -> Callable:
    """A stand-in for ``fn`` that runs the MCP tool of the same name."""
    hints = typing.get_type_hints(fn)
    returns = hints.get("return")
    params = list(inspect.signature(fn).parameters)

    def call(*args, **kwargs):
        bound = dict(zip(params, args)) | kwargs
        arguments = {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in bound.items()}
        data = tools.call(fn.__name__, arguments)
        if isinstance(returns, type) and issubclass(returns, BaseModel):
            return returns.model_validate(data.get("result", data) if set(data) == {"result"} else data)
        return data.get("result", data)

    call.__name__ = fn.__name__
    return call
