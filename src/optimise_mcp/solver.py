"""Backwards compatibility re-exports from shared.solver."""

from shared.solver import (
    HighsLPSolver,
    HighsSolverResult,
    MissingDependencyError,
    _call_solver_name_getter,
    _normalize_name,
    _resolve_name,
)

__all__ = [
    "HighsLPSolver",
    "HighsSolverResult",
    "MissingDependencyError",
    "_call_solver_name_getter",
    "_normalize_name",
    "_resolve_name",
]

