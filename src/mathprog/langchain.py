"""LangChain tool definitions for LP (Linear Programming)."""

from __future__ import annotations

from typing import Optional

from langchain.tools import tool
from mathprog.solver import LPSolver
from mathprog.validator import validate_lp_text


@tool
def validate_lp(lp_code: str) -> str:
    """
    Validate Pyomo LP (.lp) text for structural issues without attempting to solve it.

    Checks for bracket matching, required keywords, and basic syntax.
    Use this before solve_lp to diagnose issues.

    Args:
        lp_code: Pyomo LP model text.

    Returns:
        String result indicating whether the model is valid or listing validation issues.
    """
    if not lp_code or not lp_code.strip():
        return "ERROR: lp_code is required"

    validation_issues = validate_lp_text(lp_code)
    if validation_issues:
        issue_lines = [
            "LP validation found the following issues:",
            *[
                f"  Line {issue.line_number}: {issue.message} | {issue.line_content.strip()}"
                for issue in validation_issues
            ],
        ]
        return "\n".join(issue_lines)

    return "LP model is valid."


@tool
def solve_lp(
    lp_code: str,
    solver_backend: str = "glpk",
    time_limit: Optional[float] = None,
) -> str:
    """
    Solve linear/integer programming problems using Pyomo.

    Input is Pyomo LP (.lp) text. Output includes solver status, objective value,
    variable assignments, and a summary.

    Args:
        lp_code: Pyomo LP model text.
        solver_backend: LP solver backend (default: glpk).
            Try 'glpk', 'ipopt', 'cbc', or 'scip' depending on availability.
        time_limit: Solver time limit in seconds (optional).

    Returns:
        String containing solver status, objective value, and decision variable assignments.
    """
    if not lp_code or not lp_code.strip():
        return "ERROR: lp_code is required"

    # Validate before solving
    validation_issues = validate_lp_text(lp_code)
    if validation_issues:
        issue_lines = [
            "LP validation failed. Fix the issues below before calling solve_lp:",
            *[
                f"  Line {issue.line_number}: {issue.message} | {issue.line_content.strip()}"
                for issue in validation_issues
            ],
        ]
        return "\n".join(issue_lines)

    try:
        solver = LPSolver(solver_backend=solver_backend)
        payload = solver.solve(lp_code, time_limit=time_limit)
        return payload["summary"]
    except Exception as exc:
        return f"ERROR: LP solver failed: {exc}"
