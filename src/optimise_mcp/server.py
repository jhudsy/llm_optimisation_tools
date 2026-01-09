from __future__ import annotations

"""Model Context Protocol server focused on LP solve workflows."""

import argparse
import asyncio
import contextlib
import logging
import threading
from typing import Any, Callable, Optional

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from . import __version__
from shared.lp_validator import LPValidationIssue, validate_lp_text
from shared.solver import HighsLPSolver, MissingDependencyError

LOGGER = logging.getLogger("optimise_mcp.server")

INSTRUCTIONS = """Use optimise-mcp to translate natural-language optimisation briefs into LP (.lp) text and solve them with HiGHS. Workflow: (1) restate the problem; (2) craft a structured version of the problem (see format below); (3) call solve_lp exactly once with that structured version; (4) report solver status, objective value, and key decision variables from the structured payload.

- Objective: Begin with "Minimize" or "Maximize" followed by a name and colon, and then the objective function expression. E.g., "Minimize obj: 3 x1 + 4 x2". You may optionally append a single quadratic portion written as `[ ... ]/2`, e.g., `obj: a + b + [ a^2 + 4 a * b + 7 b^2 ]/2`. All squared terms or products must stay inside that bracketed block; outside of it the objective remains purely linear.
- Constraints: Introduced by "Subject To". Each constraint appears on its own line and has a name followed by a colon, the linear expression, a relational operator (<=, >=, =), and the right-hand side value *which must be a single number*. The linear expression only contains variables and their coefficients E.g., "c1: 0.333 x1 + 2 x2 <= 10". *The division and multiplication symbols (`/` and `*`) cannot be used anywhere outside the quadratic objective block* and must therefore *not* appear in constraints; rather than writing "x/2 + 3*y <= 10", write "0.5 x + 3 y <= 10".
- Variable Bounds: Introduced by "Bounds". Each bound appears on its own line and specifies the variable name, relational operator, and bound value. E.g., "0 <= x1 <= 5". Upper and lower bounds can be enetered separately, e.g., "5 <= x3". This section is optional; ommitted bounds are assumed to be non-negative.
- Variable Types: Introduced by "Generals" for integer variables and "Binaries" for binary (0-1) variables. Each variable name appears on its own line. This section is optional; ommitted variables are assumed to be continuous. It must follow the "Bounds" section if present.
- End: The model ends with the line "End".

Example complete model:
Maximize
 obj: x1 + 2 x2 + 3 x3 + x4
Subject To
 c1: - x1 + x2 + x3 + 10 x4 <= 20
 c2: x1 - 3 x2 + x3 <= 30
 c3: x2 - 3.5 x4 = 0
Bounds
 0 <= x1 <= 40
 2 <= x4 <= 3
General
 x4
End

Ensure that the model is complete and valid before calling solve_lp.
"""


SOLVE_LP_DESCRIPTION = "This tool performs optimisation through the use of linear programming (LP) and mixed integer linear programming (MILP). Input is CPLEX .lp text. Output is the objective value, variable assignments, and solver status."

VALIDATE_LP_DESCRIPTION = "This tool validates LP (.lp) text for structural issues without attempting to solve it. It checks for non-linear operators outside the quadratic objective block, variable bounds, formatting errors, and other constraint violations. Use this before solve_lp to diagnose issues."


class SolverUnavailableError(RuntimeError):
    """Raised when all solver slots are already in use."""


class SolverQuota:
    """Bounded semaphore guarding HiGHS solver instantiation."""

    def __init__(self, max_parallel: int) -> None:
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        self.max_parallel = max_parallel
        self._semaphore = threading.BoundedSemaphore(max_parallel)

    @contextlib.contextmanager
    def claim(self) -> Any:
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            raise SolverUnavailableError(
                "Solver resource temporarily unavailable. Try again in a few seconds."
            )
        try:
            yield
        finally:
            self._semaphore.release()


class OptimiseMCPHandler:
    """Business logic for the solve_lp and validate_lp tools."""

    def __init__(
        self,
        *,
        max_parallel_solvers: int = 5,
        solver_factory: Optional[Callable[[], HighsLPSolver]] = None,
        instructions: str | None = None,
    ) -> None:
        self.instructions = instructions or INSTRUCTIONS
        self._solver_quota = SolverQuota(max_parallel_solvers)
        self._solver_factory = solver_factory or HighsLPSolver

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
            content="LP is valid.",
            structured_content={"valid": True, "issues": []},
        )

    def solve_lp(self, lp_code: str, time_limit: float | None = None) -> ToolResult:
        if not lp_code or not lp_code.strip():
            raise ValueError("lp_code is required")

        LOGGER.info("Tool call received: solve_lp (time_limit=%s)", time_limit)

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
            with self._solver_quota.claim():
                solver = self._solver_factory()
                payload = solver.solve(lp_code, time_limit=time_limit)
        except SolverUnavailableError as exc:
            message = str(exc)
            LOGGER.warning("%s", message)
            return ToolResult(content=message, structured_content={"error": message})
        except MissingDependencyError as exc:
            message = f"Missing solver dependency: {exc}"
            LOGGER.error("%s", message)
            return ToolResult(content=message, structured_content={"error": str(exc)})
        except Exception as exc:  # pragma: no cover - push errors to the LLM
            message = f"HiGHS execution failed: {exc}"
            LOGGER.exception("HiGHS execution failed")
            return ToolResult(content=message, structured_content={"error": str(exc)})

        LOGGER.info("Tool solve_lp completed successfully")
        return ToolResult(content=payload["summary"], structured_content=payload)


def build_fastmcp_server(handler: OptimiseMCPHandler) -> FastMCP:
    server = FastMCP(
        name="optimise-mcp",
        version=__version__,
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
    parser = argparse.ArgumentParser(description="optimise-mcp server")
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
    parser.add_argument("--http-host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--http-port", type=int, default=8765, help="HTTP bind port")
    parser.add_argument(
        "--max-solvers",
        type=int,
        default=5,
        help="Maximum number of concurrent HiGHS solver instances",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args(argv)

    log_level = args.log_level.upper()
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    handler = OptimiseMCPHandler(max_parallel_solvers=args.max_solvers)
    fastmcp_server = build_fastmcp_server(handler)

    try:
        if args.http:
            LOGGER.info("Starting HTTP MCP server on %s:%s", args.http_host, args.http_port)
            asyncio.run(_serve_http(fastmcp_server, args.http_host, args.http_port, log_level))
        else:
            LOGGER.info("Starting stdio MCP server")
            asyncio.run(_serve_stdio(fastmcp_server, log_level))
    except KeyboardInterrupt:
        LOGGER.info("Server interrupted by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
