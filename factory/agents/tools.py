"""Typed tool interfaces.

A tool is a function from a pydantic input model to a JSON-serialisable
result. Its schema for the model is generated from the input model, so the
interface the agent sees and the one the code checks are the same. A
``Toolbox`` holds the tools one agent may use: the allow-list is the set of
tools in its box, so a tool outside it cannot be called whatever the prompt
says.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

MAX_RESULT_CHARS = 12_000  # a tool result longer than this is cut, and the cut is marked


class ToolError(RuntimeError):
    """An error the agent should see and can act on (bad input, missing item), rather than a crash."""


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[BaseModel], object]

    def schema(self) -> dict:
        s = self.input_model.model_json_schema()
        s.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": s}


def tool(name: str, description: str, input_model: type[BaseModel]):
    """Decorator: ``@tool("search", "...", SearchIn) def search(inp): ...``"""
    def wrap(fn):
        return Tool(name, description, input_model, fn)
    return wrap


@dataclass
class ToolResult:
    ok: bool
    content: str  # JSON text, or an error message


class Toolbox:
    def __init__(self, tools: list[Tool]):
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        self.tools = {t.name: t for t in tools}

    def subset(self, names: list[str]) -> "Toolbox":
        missing = set(names) - self.tools.keys()
        if missing:
            raise KeyError(f"unknown tools: {sorted(missing)}")
        return Toolbox([self.tools[n] for n in names])

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> ToolResult:
        t = self.tools.get(name)
        if t is None:
            return ToolResult(False, f"tool {name!r} is not available to this agent; available: {sorted(self.tools)}")
        if "_unparsed" in args:
            return ToolResult(False, f"arguments were not valid JSON: {args['_unparsed'][:200]}")
        try:
            inp = t.input_model.model_validate(args)
        except ValidationError as e:
            return ToolResult(False, f"invalid arguments for {name}: {e.errors(include_url=False)}")
        try:
            out = t.fn(inp)
        except ToolError as e:
            return ToolResult(False, str(e))
        text = out.model_dump_json() if isinstance(out, BaseModel) else json.dumps(out, default=str)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"... [cut: {len(text)} characters in full]"
        return ToolResult(True, text)
