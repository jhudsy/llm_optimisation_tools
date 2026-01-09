#!/usr/bin/env python3
"""Test LangChain LP tools directly without MCP or Ollama."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

# Add src to path for imports
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from langchain_optimise.tools import create_validate_lp_tool, create_solve_lp_tool


OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
DEFAULT_MODEL = "mistral"

SYSTEM_PROMPT = dedent(
    """
    You are an optimisation engineer. You must iterate strictly through tool calls.
    If tool access fails, reply with "Cannot comply: tool access required."
    
    Mandatory workflow:
    1. Restate the problem with decision variables, objective, and constraints.
    2. Produce a complete LP-format (.lp) model string using standard CPLEX LP syntax:
         - Begin with `Maximize obj:` (or `Minimize obj:`) on its own line.
         - Add a `Subject To` section with named constraints.
         - Include a `Bounds` section with both lower and upper bounds for every variable.
         - Finish with an `End` line.
    3. Call validate_lp to check the format.
    4. Call solve_lp with that LP text. Report solver status, objective value, and key variable assignments.
    """
).strip()

USER_PROMPT = dedent(
    """
    I have 110 acres that can grow wheat or corn. Wheat yields $40/acre profit and needs
    3 labour hours per acre. Corn yields $30/acre profit and needs 2 labour hours per
    acre. Total labour is capped at 240 hours. Maximise total profit and tell me how many
    acres of each crop to plant.
    
    Use the optimise tools to translate this brief into LP format, validate it, and solve it.
    Share each tool result before giving the final plan.
    """
).strip()

PLAN_SCHEMA = dedent(
    """
    Respond **only** with valid JSON following this schema:
    {
        "action": "call_tool" | "respond",
        "tool_name": "validate_lp" | "solve_lp",
        "arguments": { ... },
        "summary": "short reasoning for humans",
        "final_response": "complete answer"  // required when action is respond
    }
    Do not wrap the JSON in markdown fences.
    """
).strip()


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
                artifact_blob = json.dumps(observation.artifact, indent=2, ensure_ascii=False)
            except TypeError:
                artifact_blob = str(observation.artifact)
            return f"{body}\n\nartifact:\n{artifact_blob}"
        return body
    if isinstance(observation, AIMessage):
        return stringify_content(observation.content)
    return stringify_content(observation)


def format_tool_specs(tools: list[BaseTool]) -> str:
    """Render a compact JSON schema summary for each available tool."""
    specs: list[str] = []
    for tool in tools:
        schema: dict[str, Any] = {}
        args_schema = tool.args_schema
        if hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()
        elif isinstance(args_schema, dict):
            schema = args_schema
        specs.append(f"- {tool.name}: {tool.description}\n  Schema: {json.dumps(schema, ensure_ascii=False)}")
    return "\n".join(specs)


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


async def execute_tool_call(tool_map: dict[str, BaseTool], tool_call: dict) -> ToolMessage:
    """Invoke the requested tool and wrap the observation as a ToolMessage."""
    tool_name = tool_call.get("tool_name", tool_call.get("name", ""))
    tool = tool_map.get(tool_name)
    tool_call_id = tool_call.get("id", str(id(tool_call)))
    
    if tool is None:
        return ToolMessage(
            content=f"Tool {tool_name} is unavailable.",
            tool_call_id=tool_call_id,
            name="error",
        )

    payload = tool_call.get("arguments", {})
    if not isinstance(payload, dict):
        payload = {"input": payload}
    
    try:
        result = await tool.ainvoke(payload)
        if isinstance(result, ToolMessage):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            content, artifact = result
            return ToolMessage(
                stringify_content(content),
                tool_call_id=tool_call_id,
                name=tool.name,
            )
        return ToolMessage(
            stringify_content(result),
            tool_call_id=tool_call_id,
            name=tool.name,
        )
    except Exception as e:
        return ToolMessage(
            content=f"Tool execution failed: {e}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )


async def run_langchain_lp_test(
    model: str,
    temperature: float,
    verbose: bool,
) -> dict[str, Any]:
    """Run the LP test with LangChain tools directly."""
    # Create tools
    validate_tool = create_validate_lp_tool()
    solve_tool = create_solve_lp_tool(max_parallel_solvers=1)
    
    tools = [validate_tool, solve_tool]
    tool_map = {tool.name: tool for tool in tools}
    tool_catalog = format_tool_specs(tools)
    
    # Initialize LLM
    llm = ChatOllama(
        model=model,
        base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
        temperature=temperature,
    )
    
    system_blob = f"{SYSTEM_PROMPT}\n\nAvailable tools:\n{tool_catalog}\n\n{PLAN_SCHEMA}"
    messages = [
        SystemMessage(content=system_blob),
        HumanMessage(content=USER_PROMPT),
    ]
    
    if verbose:
        print("=" * 80)
        print("LANGCHAIN LP ROUNDTRIP TEST")
        print("=" * 80)
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Available tools: {', '.join(t.name for t in tools)}")
        print()
    
    results = {
        "model": model,
        "temperature": temperature,
        "turns": 0,
        "tool_calls": 0,
        "success": False,
        "final_response": None,
        "messages": [],
    }
    
    for turn in range(12):
        if verbose:
            print(f"\n[Turn {turn + 1}]")
        
        response: AIMessage = await llm.ainvoke(messages)
        results["turns"] += 1
        
        if verbose:
            print("LLM Response:")
            print(format_step_output(response))
            print()
        
        # Check if we have tool calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            # Model responded without tool calls - check for final answer
            content = stringify_content(response.content)
            results["final_response"] = content
            results["success"] = True
            if verbose:
                print("[DIALOGUE CONCLUDED]")
                print(f"Final Response: {content}")
            break
        
        # Process tool calls
        messages.append(response)
        for tool_call in response.tool_calls:
            results["tool_calls"] += 1
            tool_message = await execute_tool_call(tool_map, tool_call)
            if verbose:
                print(f"Tool Call: {tool_call.get('tool_name', 'unknown')}")
                print(f"Result: {format_step_output(tool_message)}")
                print()
            messages.append(tool_message)
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Test LangChain LP tools directly (no MCP or HTTP required)."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature for sampling (default: 0.7)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()
    
    try:
        results = await run_langchain_lp_test(
            model=args.model,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Turns: {results['turns']}")
        print(f"Tool calls: {results['tool_calls']}")
        print(f"Success: {results['success']}")
        print(f"Final Response: {results['final_response'][:100] if results['final_response'] else 'None'}...")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
