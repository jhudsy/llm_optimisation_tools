"""LangChain tool definitions for MiniZinc constraint programming."""

from __future__ import annotations

from typing import Optional

from langchain.tools import tool
from mzn import MiniZincSolver, validate_minizinc_text


@tool
def validate_minizinc(mzn_code: str) -> str:
    """
    Validate MiniZinc (.mzn) text for structural issues without attempting to solve it.

    Checks for bracket matching, required keywords, and basic syntax.
    Use this before solve_minizinc to diagnose issues.

    Args:
        mzn_code: MiniZinc model text.

    Returns:
        String result indicating whether the model is valid or listing validation issues.
    """
    if not mzn_code or not mzn_code.strip():
        return "ERROR: mzn_code is required"

    validation_issues = validate_minizinc_text(mzn_code)
    if validation_issues:
        issue_lines = [
            "MiniZinc validation found the following issues:",
            *[
                f"  Line {issue.line_number}: {issue.message} | {issue.line_content.strip()}"
                for issue in validation_issues
            ],
        ]
        return "\n".join(issue_lines)

    return "MiniZinc model is valid."


@tool
def solve_minizinc(
    mzn_code: str,
    solver_backend: str = "highs",
    time_limit: Optional[float] = None,
) -> str:
    """
    Solve constraint programming problems using MiniZinc.

    Input is MiniZinc (.mzn) text. Output includes solver status, objective value,
    variable assignments, and a summary.

    Args:
        mzn_code: MiniZinc model text.
        solver_backend: MiniZinc solver backend (default: highs).
            Try 'highs', 'gurobi', or 'cplex' depending on availability.
        time_limit: Solver time limit in seconds (optional).

    Returns:
        String containing solver status, objective value, and decision variable assignments.
    """
    if not mzn_code or not mzn_code.strip():
        return "ERROR: mzn_code is required"

    # Validate before solving
    validation_issues = validate_minizinc_text(mzn_code)
    if validation_issues:
        issue_lines = [
            "MiniZinc validation failed. Fix the issues below before calling solve_minizinc:",
            *[
                f"  Line {issue.line_number}: {issue.message} | {issue.line_content.strip()}"
                for issue in validation_issues
            ],
        ]
        return "\n".join(issue_lines)

    try:
        solver = MiniZincSolver(solver_backend=solver_backend)
        payload = solver.solve(mzn_code, time_limit=time_limit)
        return payload["summary"]
    except Exception as exc:
        return f"ERROR: MiniZinc solver failed: {exc}"
