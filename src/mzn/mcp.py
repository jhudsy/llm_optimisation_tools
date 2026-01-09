"""MCP server for MiniZinc constraint programming problems."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Callable, Any

import yaml
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from mzn import (
    MiniZincSolver,
    MiniZincValidationIssue,
    validate_minizinc_text,
    MissingDependencyError,
)


LOGGER = logging.getLogger("minizinc_mcp.server")

INSTRUCTIONS = """Use minizinc-mcp to translate natural-language constraint programming briefs into MiniZinc (.mzn) text and solve them with MiniZinc. Workflow: (1) restate the problem; (2) craft a structured MiniZinc model; (3) call solve_minizinc exactly once with that model; (4) report solver status, objective value, and key decision variables from the structured payload.

MiniZinc Format Guide:

- Variable Declaration: Declare decision variables with `var TYPE: varname;` where TYPE is int, float, bool, or constrained ranges.
  Examples: `var 0..100: x;` or `var float: profit;`
- Constraints: Use `constraint expr;` to define constraints. Use standard operators: +, -, *, /, =, <=, >=, <, >, etc.
  Example: `constraint 2*x + y <= 100;`
- Objective: Optionally specify optimization direction with `solve minimize expr;` or `solve maximize expr;`
  Example: `solve maximize 3*x + 2*y;`
- Output: Optionally specify output format with `output [expr];`
  Example: `output ["x=", x, ", y=", y];`

Complete MiniZinc Example:
var 0..40: wheat;
var 0..40: corn;
var float: profit;

constraint profit = 3*wheat + 2*corn;
constraint 2*wheat + corn <= 100;
constraint wheat + 2*corn <= 80;

solve maximize profit;

output ["wheat=", wheat, ", corn=", corn, ", profit=", profit];

Ensure that the model is syntactically valid before calling solve_minizinc.
"""

SOLVE_MZN_DESCRIPTION = (
    "Solves constraint programming problems using MiniZinc. Input is MiniZinc (.mzn) text. "
    "Output includes solver status, objective value, variable assignments, and summary."
)

VALIDATE_MZN_DESCRIPTION = (
    "Validates MiniZinc (.mzn) text for structural issues without attempting to solve it. "
    "Checks for bracket matching, required keywords, and basic syntax. "
    "Use this before solve_minizinc to diagnose issues."
)


class OptimiseMiniZincHandler:
    """Business logic for MiniZinc MCP tools."""

    def __init__(
        self,
        *,
        solver_backend: str = "highs",
        instructions: str | None = None,
    ) -> None:
        self.instructions = instructions or INSTRUCTIONS
        self.solver_backend = solver_backend

    def validate_minizinc(self, mzn_code: str) -> ToolResult:
        """Validate MiniZinc text and return structured issues."""
        if not mzn_code or not mzn_code.strip():
            raise ValueError("mzn_code is required")

        LOGGER.info("Tool call received: validate_minizinc")

        validation_issues = validate_minizinc_text(mzn_code)
        if validation_issues:
            LOGGER.warning("MiniZinc validation found %s issue(s)", len(validation_issues))
            issue_lines = [
                "MiniZinc validation found the following issues:",
                *[
                    f"  Line {issue.line_number}: {issue.message} | {issue.line_content.strip()}"
                    for issue in validation_issues
                ],
            ]
            return ToolResult(
                content="\n".join(issue_lines),
                structured_content={
                    "valid": False,
                    "issues": [issue.as_dict() for issue in validation_issues],
                },
            )

        LOGGER.info("Tool validate_minizinc: MiniZinc is valid")
        return ToolResult(
            content="MiniZinc model is valid.",
            structured_content={"valid": True, "issues": []},
        )

    def solve_minizinc(self, mzn_code: str, time_limit: float | None = None) -> ToolResult:
        """Solve a MiniZinc model."""
        if not mzn_code or not mzn_code.strip():
            raise ValueError("mzn_code is required")

        LOGGER.info("Tool call received: solve_minizinc (time_limit=%s)", time_limit)

        # Validate before solving
        validation_issues = validate_minizinc_text(mzn_code)
        if validation_issues:
            LOGGER.warning("MiniZinc validation failed with %s issue(s)", len(validation_issues))
            issue_lines = [
                "MiniZinc validation failed. Fix the issues below before calling solve_minizinc:",
                *[
                    f"  Line {issue.line_number}: {issue.message} | {issue.line_content.strip()}"
                    for issue in validation_issues
                ],
            ]
            return ToolResult(
                content="\n".join(issue_lines),
                structured_content={
                    "error": "MiniZinc validation failed",
                    "issues": [issue.as_dict() for issue in validation_issues],
                },
            )

        try:
            solver = MiniZincSolver(solver_backend=self.solver_backend)
            payload = solver.solve(mzn_code, time_limit=time_limit)
        except MissingDependencyError as exc:
            message = f"Missing solver dependency: {exc}"
            LOGGER.error("%s", message)
            return ToolResult(content=message, structured_content={"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            message = f"MiniZinc solver failed: {exc}"
            LOGGER.exception("MiniZinc solver failed")
            return ToolResult(content=message, structured_content={"error": str(exc)})

        LOGGER.info("Tool solve_minizinc completed successfully")
        return ToolResult(content=payload["summary"], structured_content=payload)


def build_fastmcp_server(handler: OptimiseMiniZincHandler) -> FastMCP:
    """Build FastMCP server with MiniZinc tools."""
    server = FastMCP(
        name="minizinc-mcp",
        instructions=handler.instructions,
    )

    @server.tool(name="validate_minizinc", description=VALIDATE_MZN_DESCRIPTION)
    def validate_minizinc_tool(mzn_code: str) -> ToolResult:
        return handler.validate_minizinc(mzn_code=mzn_code)

    @server.tool(name="solve_minizinc", description=SOLVE_MZN_DESCRIPTION)
    def solve_minizinc_tool(mzn_code: str, time_limit: float | None = None) -> ToolResult:
        return handler.solve_minizinc(mzn_code=mzn_code, time_limit=time_limit)

    return server


async def _serve_stdio(server: FastMCP, log_level: str) -> None:
    await server.run_stdio_async(show_banner=False, log_level=log_level)


async def _serve_http(server: FastMCP, host: str, port: int, log_level: str) -> None:
    await server.run_http_async(
        transport="streamable-http",
        host=host,
        port=port,
        path="/mcp",
        show_banner=False,
        log_level=log_level,
    )


def run(argv: Optional[list[str]] = None) -> int:
    """Run the MiniZinc MCP server."""
    parser = argparse.ArgumentParser(description="minizinc-mcp server")
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--stdio",
        action="store_true",
        help="Run using stdio transport (default)",
    )
    transport_group.add_argument(
        "--http",
        action="store_true",
        help="Run using the FastMCP streamable HTTP transport",
    )
    parser.add_argument("--http-host", help="HTTP bind host (overrides config)")
    parser.add_argument("--http-port", type=int, help="HTTP bind port (overrides config)")
    parser.add_argument(
        "--solver-backend",
        help="MiniZinc solver backend (overrides config)",
    )
    parser.add_argument("--log-level", help="Python logging level (overrides config)")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml (default: repo_root/config.yaml)",
    )
    args = parser.parse_args(argv)

    # Load config file
    config = {}
    config_path = args.config
    if not config_path:
        # Default to repo root config.yaml
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "config.yaml"
    
    if Path(config_path).exists():
        with open(config_path) as f:
            full_config = yaml.safe_load(f) or {}
            config = full_config.get("mzn", {})
            LOGGER.info(f"Loaded config from {config_path}")
    else:
        LOGGER.warning(f"Config file not found: {config_path}, using defaults")

    # Extract config values with CLI overrides
    solver_config = config.get("solver", {})
    mcp_config = config.get("mcp_server", {})
    
    solver_backend = args.solver_backend or solver_config.get("backend", "coinbc")
    http_host = args.http_host or mcp_config.get("http_host", "127.0.0.1")
    http_port = args.http_port or mcp_config.get("http_port", 8766)
    log_level = (args.log_level or mcp_config.get("log_level", "INFO")).upper()

    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    handler = OptimiseMiniZincHandler(solver_backend=solver_backend)
    fastmcp_server = build_fastmcp_server(handler)

    try:
        if args.http:
            LOGGER.info("Starting HTTP MCP server on %s:%s (backend=%s)", http_host, http_port, solver_backend)
            asyncio.run(_serve_http(fastmcp_server, http_host, http_port, log_level))
        else:
            LOGGER.info("Starting stdio MCP server (backend=%s)", solver_backend)
            asyncio.run(_serve_stdio(fastmcp_server, log_level))
    except KeyboardInterrupt:
        LOGGER.info("Server interrupted by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
