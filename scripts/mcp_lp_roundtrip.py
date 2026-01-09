#!/usr/bin/env python3
"""Run the canonical farming prompt via MCP (stdio or HTTP); defaults to Ollama as the LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from asyncio.subprocess import Process
from contextlib import asynccontextmanager
from pathlib import Path
from textwrap import dedent
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel

TransportKind = Literal["stdio", "http"]

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
HTTP_HOST = os.environ.get("OPTIMISE_MCP_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("OPTIMISE_MCP_HTTP_PORT", "8765"))

SYSTEM_PROMPT = dedent(
        """
         You are an optimisation engineer working with an MCP server named optimise-mcp.
         You must iterate strictly through MCP tool calls; prose descriptions of tools or
         pseudo-code are not acceptable. If tool access fails, reply with "Cannot comply:
         tool access required." Mandatory workflow:
         1. Restate the problem with decision variables, objective, and constraints.
         2. Produce a complete LP-format (.lp) model string using standard CPLEX LP syntax:
                - Begin with `Maximize obj:` (or `Minimize obj:`) on its own line.
                - Add a `Subject To` section with named constraints, e.g. `labor: 3 wheat + 2 corn <= 240`.
                - Include a `Bounds` section that spells out both lower and upper bounds for every variable
                    using `0 <= x <= 10` style expressions (one per line).
                - Finish with an `End` line.
                Example:
                Maximize obj:
                    40 wheat + 30 corn
                Subject To
                    land: wheat + corn <= 110
                    labor: 3 wheat + 2 corn <= 240
                Bounds
                    0 <= wheat <= 110
                    0 <= corn <= 110
                End
          3. Call solve_lp with that LP text (and optional time_limit). Report solver status,
                objective value, and the key decision variable assignments verbatim from the tool.
          Never invent other tools; the server only offers solve_lp.
        """
).strip()

USER_PROMPT = dedent(
    """
    I have 110 acres that can grow wheat or corn. Wheat yields $40/acre profit and needs
    3 labour hours per acre. Corn yields $30/acre profit and needs 2 labour hours per
    acre. Total labour is capped at 240 hours. Maximise total profit and tell me how many
    acres of each crop to plant.
    Use the optimise-mcp tools to translate this brief into LP format and solve it. Embed
    the entire .lp body inside the solve_lp call arguments. Share each tool result
    verbatim before giving the final plan—never stop after the first tool.
    """
).strip()

SOLVE_REMINDER = dedent(
    """
    You must call solve_lp with a valid LP body (Maximize/Subject To/Bounds/End) before delivering
    a final answer. Do not conclude until the solver succeeds and you have reported its status,
    objective value, and variable assignments.
    """
).strip()

PLAN_SCHEMA = dedent(
        """
        Respond **only** with valid JSON following this schema:
        {
            "action": "call_tool" | "respond",
            "tool_name": "solve_lp" (required when action is call_tool),
            "arguments": { ... }  // object matching the tool's JSON schema,
            "summary": "short reasoning for humans",
            "final_response": "complete answer"  // required when action is respond
        }
        Do not wrap the JSON in markdown fences. When action is call_tool, omit final_response.
        """
).strip()


def build_stdio_connection() -> Connection:
    """Return the stdio connection payload used by langchain-mcp-adapters."""

    repo_root = Path(__file__).resolve().parents[1]
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        raise RuntimeError("Missing .venv interpreter. Run `uv venv` or `python -m venv .venv`.")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo_root / "src"),
            "OPTIMISE_MCP_LOG_LEVEL": os.environ.get("OPTIMISE_MCP_LOG_LEVEL", "info"),
        }
    )

    return {
        "transport": "stdio",
        "command": str(python_bin),
        "args": ["-m", "optimise_mcp.server", "--stdio"],
        "env": env,
    }


def build_http_connection(host: str, port: int) -> Connection:
    """Return the HTTP connection payload for langchain-mcp-adapters."""

    return {
        "transport": "http",
        "url": f"http://{host}:{port}/mcp",
    }


def _resolve_python_command() -> tuple[str, dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[1]
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        raise RuntimeError("Missing .venv interpreter. Run `uv venv` or `python -m venv .venv`.")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo_root / "src"),
            "OPTIMISE_MCP_LOG_LEVEL": os.environ.get("OPTIMISE_MCP_LOG_LEVEL", "info"),
        }
    )
    return str(python_bin), env


async def start_http_server(host: str, port: int) -> Process:
    cmd, env = _resolve_python_command()
    return await asyncio.create_subprocess_exec(
        cmd,
        "-m",
        "optimise_mcp.server",
        "--http",
        "--http-host",
        host,
        "--http-port",
        str(port),
        env=env,
    )


async def wait_for_http_server(host: str, port: int, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            if loop.time() >= deadline:
                raise TimeoutError(f"HTTP server failed to start on {host}:{port} within {timeout}s")
            await asyncio.sleep(0.1)
            continue
        else:
            writer.close()
            await writer.wait_closed()
            return


@asynccontextmanager
async def connection_context(transport: TransportKind, host: str, port: int) -> AsyncIterator[Connection]:
    if transport == "stdio":
        yield build_stdio_connection()
        return

    server_proc = await start_http_server(host, port)
    try:
        await wait_for_http_server(host, port)
        yield build_http_connection(host, port)
    finally:
        server_proc.terminate()
        try:
            await asyncio.wait_for(server_proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            server_proc.kill()
            await server_proc.wait()


def stringify_content(content: Any) -> str:
    """Convert message and tool payloads into a readable string."""

    if content is None:
        return "<empty>"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


def format_step_output(observation: Any) -> str:
    """Normalise LangChain messages into printable strings."""

    if isinstance(observation, ToolMessage):
        body = stringify_content(observation.content)
        if observation.artifact:
            try:
                artifact_blob = json.dumps(
                    observation.artifact, indent=2, ensure_ascii=False
                )
            except TypeError:
                artifact_blob = str(observation.artifact)
            return f"{body}\n\nartifact:\n{artifact_blob}"
        return body
    if isinstance(observation, AIMessage):
        return stringify_content(observation.content)
    return stringify_content(observation)


def jsonify(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except TypeError:
        return str(payload)


def format_tool_specs(tools: list[BaseTool]) -> str:
    """Render a compact JSON schema summary for each available tool."""

    specs: list[str] = []
    for tool in tools:
        schema: dict[str, Any] = {}
        args_schema = tool.args_schema
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            schema = args_schema.model_json_schema()
        elif isinstance(args_schema, dict):
            schema = args_schema
        specs.append(
            f"- {tool.name}: {tool.description}\n  Schema: {json.dumps(schema, ensure_ascii=False)}"
        )
    return "\n".join(specs)


def validate_lp_body(lp_text: str) -> tuple[bool, str | None]:
    """Check for the key sections required by the LP reader before invoking solve_lp."""

    stripped = (lp_text or "").strip()
    if not stripped:
        return False, "LP payload is empty. Provide the full .lp text."
    lower_blob = stripped.lower()
    if "maximize" not in lower_blob and "minimize" not in lower_blob:
        return False, "LP must start with a Maximize/Minimize section (e.g. 'Maximize obj: ...')."
    if "subject to" not in lower_blob and "such that" not in lower_blob:
        return False, "Include a 'Subject To' section naming each constraint."
    if "bounds" not in lower_blob:
        return False, "Include a 'Bounds' section enumerating lower/upper limits for every variable."
    if not lower_blob.rstrip().endswith("end"):
        return False, "Terminate the LP with an 'End' line."
    return True, None


def extract_json_plan(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""

    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            start = candidate.index("{")
            end = candidate.rindex("}")
        except ValueError:
            return None
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None


async def execute_tool_call(
    tool_map: dict[str, BaseTool], tool_call: ToolCall
) -> ToolMessage:
    """Invoke the requested tool and wrap the observation as a ToolMessage."""

    tool = tool_map.get(tool_call.get("name", ""))
    tool_call_id = tool_call.get("id")
    if tool is None:
        return ToolMessage(
            content=f"Tool {tool_call.get('name')} is unavailable.",
            tool_call_id=tool_call_id,
            name="error",
            status="error",
        )

    payload = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
    if not isinstance(payload, dict):
        payload = {"input": payload}
    result = await tool.ainvoke(payload)
    if isinstance(result, ToolMessage):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
        return ToolMessage(
            stringify_content(content),
            tool_call_id=tool_call_id,
            name=tool.name,
            status="success",
            artifact=artifact,
        )
    return ToolMessage(
        stringify_content(result),
        tool_call_id=tool_call_id,
        name=tool.name,
        status="success",
    )


async def run_roundtrip(
    model: str,
    temperature: float,
    verbose: bool,
    transport: TransportKind,
) -> dict[str, Any]:
    async with connection_context(transport, HTTP_HOST, HTTP_PORT) as connection:
        client = MultiServerMCPClient({"optimise-mcp": connection})

        if verbose:
            if transport == "http":
                print(f"Connecting to optimise-mcp via HTTP on {HTTP_HOST}:{HTTP_PORT}...")
            else:
                print("Connecting to optimise-mcp via stdio...")

        async with client.session("optimise-mcp") as session:
            if verbose:
                print("Initialising MCP session and loading tools...")
            tools = await load_mcp_tools(
                session,
                callbacks=client.callbacks,
                server_name="optimise-mcp",
            )
            if verbose:
                print(f"Loaded {len(tools)} tools from optimise-mcp.")
            tool_map = {tool.name: tool for tool in tools}
            tool_catalog = format_tool_specs(tools)
            llm = ChatOllama(
                model=model,
                base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
                temperature=temperature,
            )

            system_blob = f"{SYSTEM_PROMPT}\n\nAvailable tools:\n{tool_catalog}\n\n{PLAN_SCHEMA}"
            messages: list[BaseMessage] = [
                SystemMessage(content=system_blob),
                HumanMessage(content=USER_PROMPT),
            ]
            intermediate_steps: list[tuple[ToolCall, ToolMessage]] = []
            solve_lp_seen = False
            invalid_attempts = 0

            if verbose:
                print("Starting dialogue with Ollama...\n")

            for turn in range(12):
                response: AIMessage = await llm.ainvoke(messages)
                if verbose:
                    print(f"[LLM turn {turn + 1}]")
                    print(format_step_output(response))
                    print()

                content_text = stringify_content(response.content)
                plan = extract_json_plan(content_text)
                if plan is None or "action" not in plan:
                    invalid_attempts += 1
                    messages.append(response)
                    if invalid_attempts >= 3:
                        return {
                            "output": content_text,
                            "intermediate_steps": intermediate_steps,
                            "final_message": response,
                            "solve_lp_seen": solve_lp_seen,
                        }
                    messages.append(
                        HumanMessage(
                            content=(
                                "Your previous reply was not valid JSON. Re-read the schema and reply"
                                " with JSON only."
                            )
                        )
                    )
                    continue

                invalid_attempts = 0
                action = str(plan.get("action", "")).lower()
                messages.append(response)

                if action == "call_tool":
                    tool_name = plan.get("tool_name")
                    if not tool_name or tool_name not in tool_map:
                        messages.append(
                            HumanMessage(
                                content=(
                                    f"Tool '{tool_name}' is unavailable. Choose one of {list(tool_map)}."
                                )
                            )
                        )
                        continue
                    arguments = plan.get("arguments") or {}
                    if not isinstance(arguments, dict):
                        messages.append(
                            HumanMessage(
                                content="Tool arguments must be a JSON object matching the tool schema."
                            )
                        )
                        continue
                    if tool_name == "solve_lp":
                        if solve_lp_seen:
                            messages.append(
                                HumanMessage(
                                    content=(
                                        "You already solved the LP successfully. Send a final respond action summarising"
                                        " the objective value and decision variables instead of calling solve_lp again."
                                    )
                                )
                            )
                            continue
                        lp_body = arguments.get("lp_code")
                        if not isinstance(lp_body, str) or not lp_body.strip():
                            messages.append(
                                HumanMessage(
                                    content=(
                                        "solve_lp requires a non-empty 'lp_code' string containing the full LP body."
                                        " Embed the entire .lp content directly in the arguments."
                                    )
                                )
                            )
                            continue
                        lp_ok, lp_error = validate_lp_body(lp_body)
                        if not lp_ok:
                            messages.append(
                                HumanMessage(
                                    content=(
                                        "The provided lp_code is not valid LP syntax: "
                                        f"{lp_error}. Follow the Maximize/Subject To/Bounds/End template and try again."
                                    )
                                )
                            )
                            continue
                    tool_call: ToolCall = {
                        "id": str(uuid4()),
                        "name": tool_name,
                        "args": arguments,
                    }
                    observation = await execute_tool_call(tool_map, tool_call)
                    intermediate_steps.append((tool_call, observation))
                    messages.append(observation)
                    if verbose:
                        print(f"[controller] Executed tool {tool_name}.")
                        print(f"[controller] Observation raw: {observation!r}")
                        print(f"[controller] Observation content: {format_step_output(observation)}")
                        print(
                            f"[controller] Observation extras: {getattr(observation, 'additional_kwargs', None)}"
                        )
                        print(
                            f"[controller] Observation metadata: {getattr(observation, 'response_metadata', None)}"
                        )
                    artifact = getattr(observation, "artifact", {})
                    structured_payload = None
                    if isinstance(artifact, dict):
                        structured_payload = artifact.get("structured_content")
                        if structured_payload is None:
                            structured_payload = artifact.get("structuredContent")
                    if verbose and artifact:
                        print(f"[controller] Observation artifact: {artifact}")
                    if tool_name == "solve_lp" and isinstance(artifact, dict):
                        if structured_payload:
                            messages.append(
                                HumanMessage(
                                    content="Solver output:\n" + jsonify(structured_payload)
                                )
                            )
                    if (
                        tool_name == "solve_lp"
                        and getattr(observation, "status", None) != "error"
                    ):
                        solve_lp_seen = True
                        messages.append(
                            HumanMessage(
                                content=(
                                    "solve_lp succeeded. Now emit a JSON plan with action 'respond' that summarises the"
                                    " solver status, objective value, and acreage decisions."
                                )
                            )
                        )
                    continue

                if action == "respond":
                    if not solve_lp_seen:
                        messages.append(HumanMessage(content=SOLVE_REMINDER))
                        continue
                    final_text = (
                        plan.get("final_response")
                        or plan.get("summary")
                        or content_text
                    )
                    return {
                        "output": final_text,
                        "intermediate_steps": intermediate_steps,
                        "final_message": response,
                        "solve_lp_seen": True,
                    }

                messages.append(
                    HumanMessage(content="Unknown action. Use 'call_tool' or 'respond' only.")
                )

    raise RuntimeError("LLM stopped without emitting a final answer.")


def render_result(result: dict[str, Any]) -> None:
    steps = result.get("intermediate_steps", [])
    if steps:
        print("---- Tool Calls ----")
        for idx, (tool_call, observation) in enumerate(steps, start=1):
            tool_name = tool_call.get("name", "<unknown>")
            print(f"[{idx}] {tool_name}")
            print("Arguments:")
            print(jsonify(tool_call.get("args", {})))
            print("Observation:")
            print(format_step_output(observation))
            print()
    else:
        print("No tool calls captured.")

    print("---- Final Answer ----")
    print(result.get("output", "<no output>"))

    final_message = result.get("final_message")
    if isinstance(final_message, AIMessage):
        metadata = getattr(final_message, "response_metadata", None)
        scalar_metadata = {
            key: value
            for key, value in (metadata or {}).items()
            if not isinstance(value, (dict, list))
        }
        if scalar_metadata:
            print("\n---- Ollama Metadata ----")
            for key, value in scalar_metadata.items():
                print(f"{key}: {value}")

    if not result.get("solve_lp_seen", False):
        print("\nWarning: no successful solve_lp call was observed.")


def main(
    argv: list[str] | None = None,
    *,
    default_transport: TransportKind = "stdio",
    force_transport: TransportKind | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature forwarded to ChatOllama.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable LangChain agent tracing output.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=default_transport,
        help="MCP transport to use (stdio launches optimise-mcp over pipes, http starts the FastMCP server).",
    )
    args = parser.parse_args(argv)
    transport: TransportKind = force_transport or args.transport  # type: ignore[assignment]

    try:
        result = asyncio.run(run_roundtrip(args.model, args.temperature, args.verbose, transport))
    except KeyboardInterrupt:
        return 130

    render_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
