"""MCP server for LP (Linear Programming) problems."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from mathprog.solver import HighsLPSolver
from mathprog.validator import validate_lp_text


LOGGER = logging.getLogger("lp_mcp.server")

INSTRUCTIONS = """Use lp-mcp to translate natural-language optimization briefs into Pyomo LP (.lp) text and solve them with Pyomo. Workflow: (1) restate the problem; (2) craft a structured LP model; (3) call solve_lp exactly once with that model; (4) report solver status, objective value, and key decision variables from the structured payload.

Pyomo LP Format Guide:

- Variable Declaration: Declare decision variables with `var varname >= lower, <= upper;`
  Examples: `var x >= 0;`, `var y >= 0, <= 100;`
- Constraint Declaration: Use `subject to { constraint_name: expression; }`
  Example: `subject to { capacity: 2*x + y <= 100; }`
- Objective: Specify optimization direction with `maximize: expr;` or `minimize: expr;`
  Example: `maximize: 3*x + 2*y;`

Complete LP Example:
var x >= 0;
var y >= 0;
maximize: 3*x + 2*y;
subject to {
  c1: 2*x + y <= 100;
  c2: x + 2*y <= 80;
}

Ensure that the model is syntactically valid before calling solve_lp.
"""

SOLVE_LP_DESCRIPTION = (
    "Solves linear/integer programming problems using Pyomo. Input is Pyomo LP (.lp) text. "
    "Output includes solver status, objective value, variable assignments, and summary."
)

VALIDATE_LP_DESCRIPTION = (
    "Validates Pyomo LP (.lp) text for structural issues without attempting to solve it. "
    "Checks for bracket matching, required keywords, and basic syntax. "
    "Use this before solve_lp to diagnose issues."
)


class OptimiseLPHandler:
    """Business logic for LP MCP tools."""

    def __init__(
        self,
        *,
        solver_backend: str = "glpk",
        instructions: str | None = None,
    ) -> None:
        self.instructions = instructions or INSTRUCTIONS
        self.solver_backend = solver_backend

    def validate_lp(self, lp_code: str) -> ToolResult:
        """Validate LP text and return structured issues."""
        if not lp_code or not lp_code.strip():
            raise ValueError("lp_code is required")

        LOGGER.info("Tool call received: validate_lp")

        validation_issues = validate_lp_text(lp_code)
        if validation_issues:
            LOGGER.warning("LP validation found %s issue(s)", len(validation_issues))
            issue_lines = [
                "LP validation found the following issues:",
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

        LOGGER.info("Tool validate_lp: LP is valid")
        return ToolResult(
            content="LP model is valid.",
            structured_content={"valid": True, "issues": []},
        )

    def solve_lp(self, lp_code: str, time_limit: float | None = None) -> ToolResult:
        """Solve an LP model."""
        if not lp_code or not lp_code.strip():
            raise ValueError("lp_code is required")

        LOGGER.info("Tool call received: solve_lp (time_limit=%s)", time_limit)

        # Validate before solving
        validation_issues = validate_lp_text(lp_code)
        if validation_issues:
            LOGGER.warning("LP validation failed with %s issue(s)", len(validation_issues))
            issue_lines = [
                "LP validation failed. Fix the issues below before calling solve_lp:",
                *[
                    f"  Line {issue.line_number}: {issue.message} | {issue.line_content.strip()}"
                    for issue in validation_issues
                ],
            ]
            return ToolResult(
                content="\n".join(issue_lines),
                structured_content={
                    "error": "LP validation failed",
                    "issues": [issue.as_dict() for issue in validation_issues],
                },
            )

        try:
            solver = HighsLPSolver(solver_backend=self.solver_backend)
            payload = solver.solve(lp_code, time_limit=time_limit)
        except Exception as exc:  # pragma: no cover
            message = f"LP solver failed: {exc}"
            LOGGER.exception("LP solver failed")
            return ToolResult(content=message, structured_content={"error": str(exc)})

        LOGGER.info("Tool solve_lp completed successfully")
        return ToolResult(content=payload["summary"], structured_content=payload)


def build_fastmcp_server(handler: OptimiseLPHandler) -> FastMCP:
    """Build FastMCP server with LP tools."""
    server = FastMCP(
        name="lp-mcp",
        instructions=handler.instructions,
    )

    @server.tool(name="validate_lp", description=VALIDATE_LP_DESCRIPTION)
    def validate_lp_tool(lp_code: str) -> ToolResult:
        return handler.validate_lp(lp_code=lp_code)

    @server.tool(name="solve_lp", description=SOLVE_LP_DESCRIPTION)
    def solve_lp_tool(lp_code: str, time_limit: float | None = None) -> ToolResult:
        return handler.solve_lp(lp_code=lp_code, time_limit=time_limit)

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
    """Run the LP MCP server."""
    parser = argparse.ArgumentParser(description="lp-mcp server")
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
        help="LP solver backend (overrides config)",
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
            config = full_config.get("mathprog", {})
            LOGGER.info(f"Loaded config from {config_path}")
    else:
        LOGGER.warning(f"Config file not found: {config_path}, using defaults")

    # Extract config values with CLI overrides
    solver_config = config.get("solver", {})
    mcp_config = config.get("mcp_server", {})
    
    solver_backend = args.solver_backend or solver_config.get("backend", "highs")
    http_host = args.http_host or mcp_config.get("http_host", "127.0.0.1")
    http_port = args.http_port or mcp_config.get("http_port", 8765)
    log_level = (args.log_level or mcp_config.get("log_level", "INFO")).upper()

    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    handler = OptimiseLPHandler(solver_backend=solver_backend)
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
