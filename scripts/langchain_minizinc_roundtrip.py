#!/usr/bin/env python3
"""Test LangChain MiniZinc tools directly without MCP or Ollama."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from langchain_optimise.minizinc_tools import create_solve_minizinc_tool, create_validate_minizinc_tool

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
DEFAULT_MODEL = "mistral"

SYSTEM_PROMPT = dedent(
    """
    You are an optimisation engineer. You must iterate strictly through tool calls.
    If tool access fails, reply with "Cannot comply: tool access required."
    
    CRITICAL: Each tool call must contain a COMPLETE, STANDALONE MiniZinc model.
    Never send partial models or fragments.
    
    Mandatory workflow:
    1. Restate the problem with decision variables, objective, and constraints.
    2. Build ONE complete MiniZinc model string with ALL components:
       - Decision variables (one per line): "var 0..110: wheat;" then "var 0..110: corn;"
       - Objective variable: "var float: profit;"
       - ALL constraints including bounds: "constraint wheat + corn <= 110;" and "constraint 3*wheat + 2*corn <= 240;"
       - Objective definition: "constraint profit = 40*wheat + 30*corn;"
       - Solve statement: "solve maximize profit;"
       - Output statement: "output [\"wheat=\", show(wheat), \" corn=\", show(corn), \" profit=\", show(profit)];"
    3. Call validate_minizinc ONCE with the complete model. If validation fails, fix and retry.
    4. After validation succeeds, the solver will run automatically.
    5. When you receive solver results, YOU MUST:
       - Extract the EXACT variable values from the solver output in the tool message
       - Copy these values VERBATIM - do NOT calculate, estimate, or invent values
       - Emit action "respond" with final_response containing the exact solver results
       - Format: "Plant X acres wheat, Y acres corn for $Z profit" using ONLY the values from the solver
    
    FORBIDDEN: Do NOT make up values, estimate results, or calculate on your own.
    REQUIRED: Copy the exact numbers from the 'variables' field in the solver tool message.
    """
).strip()

USER_PROMPT = dedent(
    """
    I have 110 acres that can grow wheat or corn. Wheat yields $40/acre profit and needs
    3 labour hours per acre. Corn yields $30/acre profit and needs 2 labour hours per
    acre. Total labour is capped at 240 hours. Maximise total profit and tell me how many
    acres of each crop to plant.
    
    Use the optimise tools to translate this brief into MiniZinc format, validate it, and solve it.
    Share each tool result before giving the final plan.
    """
).strip()

PLAN_SCHEMA = dedent(
    """
    Respond **only** with valid JSON following this schema:
    {
        "action": "call_tool" | "respond",
        "tool_name": "validate_minizinc" | "solve_minizinc",
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
        specs.append(
            f"- {tool.name}: {tool.description}\n  Schema: {json.dumps(schema, ensure_ascii=False)}"
        )
    return "\n".join(specs)


def extract_json_plan(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from model output."""
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


def extract_json_plans(text: str) -> list[dict[str, Any]]:
    """Extract all JSON objects from text, fixing common LLM JSON glitches."""
    items: list[dict[str, Any]] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    escape_next = False

    for ch in text:
        if ch == "{" and not in_string:
            depth += 1
            buf.append(ch)
        elif ch == "}" and not in_string:
            buf.append(ch)
            depth -= 1
            if depth == 0 and buf:
                candidate = "".join(buf)
                buf.clear()
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        items.append(obj)
                        continue
                except json.JSONDecodeError:
                    pass
                try:
                    fixed = (
                        candidate.replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                    )
                    obj = json.loads(fixed)
                    if isinstance(obj, dict):
                        items.append(obj)
                        continue
                except json.JSONDecodeError:
                    pass
                try:
                    fixed = (
                        candidate.replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                    )
                    fixed = re.sub(r'"\s*\+\s*"', "", fixed)
                    obj = json.loads(fixed)
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    pass
        elif depth > 0:
            if ch == "\"" and not escape_next:
                in_string = not in_string
            escape_next = ch == "\\" and not escape_next
            buf.append(ch)
    return items


async def execute_tool_call(tool_map: dict[str, BaseTool], tool_call: dict) -> ToolMessage:
    """Invoke the requested tool and normalise the result into a ToolMessage."""

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
            status = "error" if (isinstance(content, dict) and "error" in content) else "success"
            return ToolMessage(
                stringify_content(content),
                tool_call_id=tool_call_id,
                name=tool.name,
                status=status,
                artifact=artifact,
            )
        if isinstance(result, dict):
            status = "error" if "error" in result else "success"
            return ToolMessage(
                stringify_content(result),
                tool_call_id=tool_call_id,
                name=tool.name,
                status=status,
                artifact={"structured_content": result},
            )
        text = stringify_content(result)
        status = "error" if text.lower().startswith("tool execution failed") else "success"
        return ToolMessage(
            text,
            tool_call_id=tool_call_id,
            name=tool.name,
            status=status,
        )
    except Exception as e:  # pragma: no cover - defensive
        return ToolMessage(
            content=f"Tool execution failed: {e}",
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )


async def run_langchain_minizinc_test(
    model: str,
    temperature: float,
    verbose: bool,
) -> dict[str, Any]:
    """Run the MiniZinc test with LangChain tools directly."""

    validate_tool = create_validate_minizinc_tool()
    solve_tool = create_solve_minizinc_tool(max_parallel_solvers=1)

    tools = [validate_tool, solve_tool]
    tool_map = {tool.name: tool for tool in tools}
    tool_catalog = format_tool_specs(tools)

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
        print("LANGCHAIN MINIZINC ROUNDTRIP TEST")
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

    solve_minizinc_seen = False
    auto_solve_triggered_this_turn = False
    invalid_attempts = 0

    for turn in range(12):
        auto_solve_triggered_this_turn = False  # Reset at start of each turn
        
        if verbose:
            print(f"\n[Turn {turn + 1}]")

        response: AIMessage = await llm.ainvoke(messages)
        results["turns"] += 1

        if verbose:
            print("LLM Response:")
            print(format_step_output(response))
            print()

        content_text = stringify_content(response.content)
        plans = extract_json_plans(content_text)

        if not plans:
            invalid_attempts += 1
            messages.append(response)
            if invalid_attempts >= 3:
                results["final_response"] = content_text
                results["success"] = solve_minizinc_seen
                return results
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous reply was not valid JSON. Re-read the schema and reply with JSON only."
                    )
                )
            )
            continue

        invalid_attempts = 0
        messages.append(response)

        for plan in plans:
            action = str(plan.get("action", "")).lower()
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

                # Check for placeholder code patterns
                mzn_code = arguments.get("mzn_code", "")
                if isinstance(mzn_code, str) and ("<your" in mzn_code.lower() or "..." in mzn_code):
                    messages.append(
                        HumanMessage(
                            content=(
                                "Do NOT use placeholders like '<your code here>' or '...'. "
                                "Provide the COMPLETE MiniZinc model code with all declarations, constraints, solve, and output."
                            )
                        )
                    )
                    continue

                if tool_name == "solve_minizinc":
                    if solve_minizinc_seen:
                        messages.append(
                            HumanMessage(
                                content=(
                                    "You already solved the model successfully. Do NOT call any more tools. "
                                    "Emit action 'respond' (not 'call_tool') with the solver results from the previous tool message."
                                )
                            )
                        )
                        continue
                    mzn_body = arguments.get("mzn_code")
                    if not isinstance(mzn_body, str) or not mzn_body.strip():
                        messages.append(
                            HumanMessage(
                                content=(
                                    "solve_minizinc requires a non-empty 'mzn_code' string containing the full MiniZinc model. Embed the entire .mzn content directly in the arguments."
                                )
                            )
                        )
                        continue

                if tool_name == "validate_minizinc":
                    if solve_minizinc_seen:
                        messages.append(
                            HumanMessage(
                                content=(
                                    "The model has already been validated and solved. Do NOT validate again. "
                                    "Emit action 'respond' with the solver results."
                                )
                            )
                        )
                        continue

                tool_call = {
                    "id": f"tc-{turn}",
                    "name": tool_name,
                    "arguments": arguments,
                }
                tool_message = await execute_tool_call(tool_map, tool_call)
                results["tool_calls"] += 1

                if verbose:
                    print(f"Tool Call: {tool_name}")
                    print(f"Result: {format_step_output(tool_message)}")
                    print()

                messages.append(tool_message)

                if tool_name == "validate_minizinc":
                    res_dict = None
                    art = getattr(tool_message, "artifact", None)
                    if isinstance(art, dict) and isinstance(art.get("structured_content"), dict):
                        res_dict = art["structured_content"]

                    if isinstance(res_dict, dict) and res_dict.get("valid") is True:
                        mzn_body = arguments.get("mzn_code")
                        if isinstance(mzn_body, str) and mzn_body.strip():
                            auto_solve_triggered_this_turn = True  # Set flag when auto-solving
                            auto_tool_call = {
                                "id": f"tc-{turn}-auto-solve",
                                "name": "solve_minizinc",
                                "arguments": {"mzn_code": mzn_body},
                            }
                            auto_tool_message = await execute_tool_call(tool_map, auto_tool_call)
                            results["tool_calls"] += 1

                            if verbose:
                                print("Auto Tool Call: solve_minizinc")
                                print(f"Result: {format_step_output(auto_tool_message)}")
                                print()

                            messages.append(auto_tool_message)

                            if getattr(auto_tool_message, "status", None) != "error":
                                solve_minizinc_seen = True
                                # Extract actual values from solver output
                                solver_vars = ""
                                art = getattr(auto_tool_message, "artifact", None)
                                if isinstance(art, dict) and isinstance(art.get("structured_content"), dict):
                                    solver_data = art["structured_content"]
                                    variables = solver_data.get("variables", {})
                                    solver_vars = f"\nACTUAL SOLVER OUTPUT: {variables}"
                                
                                messages.append(
                                    HumanMessage(
                                        content=(
                                            f"SOLVER COMPLETED SUCCESSFULLY.{solver_vars}\n\n"
                                            "You MUST now emit EXACTLY ONE JSON with action 'respond' (NOT 'call_tool').\n\n"
                                            "CRITICAL INSTRUCTIONS:\n"
                                            "1. Look at the ACTUAL SOLVER OUTPUT above\n"
                                            "2. Extract the EXACT numbers for wheat, corn, and profit\n"
                                            "3. Copy those EXACT values into your final_response\n"
                                            "4. Do NOT calculate, estimate, or invent ANY values\n"
                                            "5. Do NOT call any tools\n\n"
                                            "Example format: 'Plant 20 acres wheat and 90 acres corn for $3500.0 profit' "
                                            "(using the actual numbers from above, not these example numbers)"
                                        )
                                    )
                                )
                            continue
                        messages.append(
                            HumanMessage(
                                content=(
                                    "Validation passed, but no 'mzn_code' was provided in the tool arguments. Include the full MiniZinc model string as 'mzn_code' and call solve_minizinc."
                                )
                            )
                        )
                        continue

                    if isinstance(res_dict, dict) and res_dict.get("valid") is False:
                        issues = res_dict.get("issues", [])
                        issues_summary = issues[:3] if isinstance(issues, list) else issues
                        messages.append(
                            HumanMessage(
                                content=(
                                    f"Validation failed. Please fix the MiniZinc model and revalidate. Detected issues (first shown): {issues_summary}"
                                )
                            )
                        )
                        continue

                if tool_name == "solve_minizinc" and getattr(tool_message, "status", None) != "error":
                    solve_minizinc_seen = True
                    messages.append(
                        HumanMessage(
                            content=(
                                "solve_minizinc succeeded. Now emit a JSON plan with action 'respond' that summarises the solver status, objective value, and decisions."
                            )
                        )
                    )

                continue

            if action == "respond":
                if not solve_minizinc_seen:
                    messages.append(
                        HumanMessage(
                            content=(
                                "You must call solve_minizinc successfully before delivering a final answer."
                            )
                        )
                    )
                    continue
                
                # Reject respond in the same turn as auto-solve - LLM hasn't seen solver results yet
                if auto_solve_triggered_this_turn:
                    if verbose:
                        print("(Ignoring 'respond' action - auto-solve triggered this turn, LLM will see results next turn)")
                    continue
                
                final_text = plan.get("final_response") or plan.get("summary") or content_text
                results["final_response"] = final_text
                results["success"] = True
                return results

            messages.append(
                HumanMessage(content="Unknown action. Use 'call_tool' or 'respond' only.")
            )

    results["final_response"] = "Dialogue ended without a final respond action."
    results["success"] = solve_minizinc_seen
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Test LangChain MiniZinc tools directly (no MCP or HTTP required)."
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
        results = await run_langchain_minizinc_test(
            model=args.model,
            temperature=args.temperature,
            verbose=args.verbose,
        )

        if "error" in results:
            print(f"Error: {results['error']}")
            return 1

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Turns: {results['turns']}")
        print(f"Tool calls: {results['tool_calls']}")
        print(f"Success: {results['success']}")
        print(f"Final Response: {results.get('final_response', 'None')}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
