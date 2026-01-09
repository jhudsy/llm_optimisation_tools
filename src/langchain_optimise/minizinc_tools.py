"""LangChain tool definitions for MiniZinc validation and solving."""

from typing import Optional, Callable, Any
import logging
import contextlib
import threading

from langchain_core.tools import tool
try:
    from pydantic import BaseModel, Field
except ImportError:
    from pydantic_v1 import BaseModel, Field  # type: ignore

from mzn.validator import validate_minizinc_text, MiniZincValidationIssue
from mzn.solver import MiniZincSolver, MissingDependencyError


LOGGER = logging.getLogger("langchain_optimise.minizinc_tools")

MZN_FORMAT_GUIDE = """
MiniZinc Format Guide:

- Variable Declaration: Declare decision variables with `var TYPE: varname;` where TYPE is int, float, 
  bool, or constrained ranges.
  Examples: `var 0..100: x;` or `var float: profit;`

- Constraints: Use `constraint expr;` to define constraints. Use standard operators: 
  +, -, *, /, =, <=, >=, <, >, etc.
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


class SolverUnavailableError(RuntimeError):
    """Raised when all solver slots are already in use."""


class SolverQuota:
    """Bounded semaphore guarding MiniZinc solver instantiation."""

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


class ValidateMiniZincInput(BaseModel):
    """Input schema for MiniZinc validation."""

    mzn_code: str = Field(description="MiniZinc model code")


class SolveMiniZincInput(BaseModel):
    """Input schema for MiniZinc solving."""

    mzn_code: str = Field(description="MiniZinc model code")
    time_limit: Optional[float] = Field(
        default=None,
        description="Optional solver time limit in seconds",
    )
    solver_backend: Optional[str] = Field(
        default="coinbc",
        description="MiniZinc solver backend (default: coinbc)",
    )


def _require_mzn_code(mzn_code: str) -> Optional[dict]:
    """Validate that MiniZinc code is provided."""
    if not mzn_code or not mzn_code.strip():
        return {
            "valid": False,
            "error": "mzn_code is required",
            "issues": [],
        }
    return None


def _format_validation_issues(issues: list) -> list:
    """Convert validation issues to dict format."""
    return [issue.as_dict() for issue in issues]


def create_validate_minizinc_tool() -> Callable:
    """Create a LangChain tool for validating MiniZinc models.

    Returns:
        A LangChain tool callable.
    """

    @tool(
        "validate_minizinc",
        args_schema=ValidateMiniZincInput,
        description=(
            "Validates MiniZinc (.mzn) text for structural issues without attempting to solve it. "
            "Checks for bracket matching, required keywords, and basic syntax. "
            "Use this before solve_minizinc to diagnose issues."
            f"\n\n{MZN_FORMAT_GUIDE}"
        ),
    )
    def validate_minizinc_tool(mzn_code: str) -> dict:
        """Validate a MiniZinc model.

        Args:
            mzn_code: MiniZinc model code

        Returns:
            Dictionary with validation result
        """
        error = _require_mzn_code(mzn_code)
        if error:
            return error

        LOGGER.info("validate_minizinc tool called")

        validation_issues = validate_minizinc_text(mzn_code)
        if validation_issues:
            LOGGER.warning("MiniZinc validation found %s issue(s)", len(validation_issues))
            return {
                "valid": False,
                "error": "MiniZinc validation failed",
                "issues": _format_validation_issues(validation_issues),
            }

        LOGGER.info("validate_minizinc completed: model is valid")
        return {"valid": True, "issues": []}

    return validate_minizinc_tool


def create_solve_minizinc_tool(
    max_parallel_solvers: int = 5,
    solver_factory: Optional[Callable[[], MiniZincSolver]] = None,
) -> Callable:
    """Create a LangChain tool for solving MiniZinc problems.

    Args:
        max_parallel_solvers: Maximum number of concurrent solver instances.
        solver_factory: Optional callable returning MiniZincSolver instances.
                       Defaults to MiniZincSolver with HiGHS backend.

    Returns:
        A LangChain tool callable.
    """
    solver_quota = SolverQuota(max_parallel_solvers)

    def default_solver_factory() -> MiniZincSolver:
        return MiniZincSolver(solver_backend="coinbc")

    solver_factory = solver_factory or default_solver_factory

    @tool(
        "solve_minizinc",
        args_schema=SolveMiniZincInput,
        description=(
            "Solves constraint programming problems using MiniZinc. Input is MiniZinc (.mzn) text. "
            "Validation is performed first; if any issues are found, they are returned "
            "without attempting to solve. On success, output includes solver status, "
            "objective value, and variable assignments."
            f"\n\n{MZN_FORMAT_GUIDE}"
        ),
    )
    def solve_minizinc_tool(
        mzn_code: str,
        time_limit: Optional[float] = None,
        solver_backend: Optional[str] = None,
    ) -> dict:
        """Solve a MiniZinc constraint programming problem.

        Args:
            mzn_code: MiniZinc model code
            time_limit: Optional solver time limit in seconds
            solver_backend: Optional solver backend (default: coinbc)

        Returns:
            Dictionary with solver results or error message.
        """
        error = _require_mzn_code(mzn_code)
        if error:
            return error

        LOGGER.info("solve_minizinc tool called (time_limit=%s, backend=%s)", time_limit, solver_backend)

        # Validate before solving
        validation_issues = validate_minizinc_text(mzn_code)
        if validation_issues:
            LOGGER.warning("MiniZinc validation failed with %s issue(s)", len(validation_issues))
            return {
                "error": "MiniZinc validation failed",
                "issues": _format_validation_issues(validation_issues),
            }

        try:
            with solver_quota.claim():
                if solver_backend:
                    solver = MiniZincSolver(solver_backend=solver_backend)
                else:
                    solver = solver_factory()
                payload = solver.solve(mzn_code, time_limit=time_limit)
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

        LOGGER.info("solve_minizinc completed successfully")
        return payload

    return solve_minizinc_tool
