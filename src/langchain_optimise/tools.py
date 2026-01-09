"""LangChain tool definitions for LP validation and solving."""

from typing import Optional, Callable, Any
import logging
import contextlib
import threading

from langchain_core.tools import tool
try:
    from pydantic import BaseModel, Field
except ImportError:
    from pydantic_v1 import BaseModel, Field  # type: ignore

from shared.lp_validator import validate_lp_text, LPValidationIssue
from shared.solver import HighsLPSolver, MissingDependencyError


LOGGER = logging.getLogger("langchain_optimise.tools")

LP_FORMAT_GUIDE = """
LP Format Guide (CPLEX .lp text):

- Objective function: Begin with "Minimize" or "Maximize" followed by a name and colon, then the objective function. 
  Example: "Minimize obj: 3 x1 + 4 x2". You may optionally include a quadratic block as `[ ... ]/2`, 
  e.g., `obj: a + b + [ a^2 + 4 a * b + 7 b^2 ]/2`. All squared terms and products MUST stay inside the 
  bracketed block; outside of it, the objective is purely linear.

- Constraints: Introduced by "Subject To". Each constraint has a name, colon, linear expression, 
  relational operator (<=, >=, =), and numeric right-hand side. 
  Example: "c1: 0.333 x1 + 2 x2 <= 10". Multiplication and division symbols (* and /) are forbidden 
  outside the quadratic block; write "0.5 x + 3 y <= 10" instead of "x/2 + 3*y <= 10".

- Variable Bounds: Introduced by "Bounds". Each line specifies a variable, relational operator, and bound.
  Examples: "0 <= x1 <= 5" or "5 <= x3". This section is optional; omitted bounds default to non-negative.

- Variable Types: Use "Generals" for integer variables or "Binaries" for 0-1 variables. 
  Each variable appears on its own line. Optional; omitted variables are continuous. 
  Must follow "Bounds" section if present.

- End: The model ends with the line "End".

Example:
Maximize
 profit: 3 x + 2 y + [ x^2 + y^2 ]/2
Subject To
 labor: 2 x + y <= 100
 material: x + 2 y <= 80
Bounds
 0 <= x <= 40
 0 <= y <= 40
Generals
 x
End
"""


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


class ValidateLPInput(BaseModel):
    """Input schema for LP validation."""

    lp_code: str = Field(description="LP code in CPLEX .lp format")


class SolveLPInput(BaseModel):
    """Input schema for LP solving."""

    lp_code: str = Field(description="LP code in CPLEX .lp format")
    time_limit: Optional[float] = Field(
        default=None,
        description="Optional solver time limit in seconds",
    )


def _require_lp_code(lp_code: str) -> Optional[dict]:
    """Validate that LP code is provided.
    
    Args:
        lp_code: The LP code to check.
        
    Returns:
        Error dict if validation fails, None otherwise.
    """
    if not lp_code or not lp_code.strip():
        return {
            "valid": False,
            "error": "lp_code is required",
            "issues": [
                {
                    "line_number": 0,
                    "message": "lp_code is required",
                    "line_content": "",
                }
            ],
        }
    return None


def _format_validation_issues(issues: list) -> list:
    """Convert validation issues to dict format.
    
    Args:
        issues: List of LPValidationIssue objects.
        
    Returns:
        List of dict representations.
    """
    return [issue.as_dict() for issue in issues]


def create_validate_lp_tool() -> Callable:
    """Create a LangChain tool for validating LP text.

    Returns:
        A LangChain tool callable.
    """

    @tool(
        "validate_lp",
        args_schema=ValidateLPInput,
        description=(
            f"Validates LP (.lp) text for structural issues without attempting to solve it. "
            f"Checks for non-linear operators outside the quadratic objective block, "
            f"variable bounds, formatting errors, and other constraint violations. "
            f"Use this before solve_lp to diagnose issues. "
            f"\n\n{LP_FORMAT_GUIDE}"
        ),
    )
    def validate_lp_tool(lp_code: str) -> dict:
        """Validate LP text and return structured issues.

        Args:
            lp_code: LP code in CPLEX .lp format

        Returns:
            Dictionary with 'valid' (bool) and 'issues' (list) keys.
        """
        error = _require_lp_code(lp_code)
        if error:
            return error

        LOGGER.info("validate_lp tool called")
        validation_issues = validate_lp_text(lp_code)

        if validation_issues:
            LOGGER.warning("LP validation found %s issue(s)", len(validation_issues))
            return {
                "valid": False,
                "issues": _format_validation_issues(validation_issues),
            }

        LOGGER.info("LP passed validation")
        return {"valid": True, "issues": []}

    return validate_lp_tool


def create_solve_lp_tool(
    max_parallel_solvers: int = 5,
    solver_factory: Optional[Callable[[], HighsLPSolver]] = None,
) -> Callable:
    """Create a LangChain tool for solving LP/MILP problems.

    Args:
        max_parallel_solvers: Maximum number of concurrent solver instances.
        solver_factory: Optional callable returning HighsLPSolver instances.
                       Defaults to HighsLPSolver (HiGHS backend).

    Returns:
        A LangChain tool callable.
    """
    solver_quota = SolverQuota(max_parallel_solvers)
    
    def default_solver_factory() -> HighsLPSolver:
        return HighsLPSolver()
    
    solver_factory = solver_factory or default_solver_factory

    @tool(
        "solve_lp",
        args_schema=SolveLPInput,
        description=(
            f"Solves LP and MILP problems using HiGHS. Input is CPLEX .lp text. "
            f"Validation is performed first; if any issues are found, they are returned "
            f"without attempting to solve. On success, output includes solver status, "
            f"objective value, primal/dual variables, and reduced costs. "
            f"\n\n{LP_FORMAT_GUIDE}"
        ),
    )
    def solve_lp_tool(lp_code: str, time_limit: Optional[float] = None) -> dict:
        """Solve an LP/MILP problem.

        Args:
            lp_code: LP code in CPLEX .lp format
            time_limit: Optional solver time limit in seconds

        Returns:
            Dictionary with solver results or error message.
        """
        error = _require_lp_code(lp_code)
        if error:
            error["error"] = "lp_code is required"
            return error

        LOGGER.info("solve_lp tool called (time_limit=%s)", time_limit)

        # Validate before solving
        validation_issues = validate_lp_text(lp_code)
        if validation_issues:
            LOGGER.warning("LP validation failed with %s issue(s)", len(validation_issues))
            return {
                "error": "LP validation failed",
                "issues": _format_validation_issues(validation_issues),
            }

        try:
            with solver_quota.claim():
                solver = solver_factory()
                payload = solver.solve(lp_code, time_limit=time_limit)
        except SolverUnavailableError as exc:
            LOGGER.warning("%s", str(exc))
            return {"error": str(exc)}
        except MissingDependencyError as exc:
            message = f"Solver dependency missing: {exc}"
            LOGGER.error("%s", message)
            return {"error": message}
        except Exception as exc:
            message = f"Solver execution failed: {exc}"
            LOGGER.exception("Solver execution failed")
            return {"error": message}

        LOGGER.info("solve_lp completed successfully")
        return payload

    return solve_lp_tool
