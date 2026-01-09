from __future__ import annotations

"""HiGHS-backed LP solver utilities."""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from highspy import Highs, HighsStatus
except ImportError:  # pragma: no cover - exercised only when HiGHS bindings are missing.
    Highs = None  # type: ignore[assignment]
    HighsStatus = None  # type: ignore[assignment]


LOGGER = logging.getLogger("shared.solver")


class MissingDependencyError(RuntimeError):
    """Raised when optional solver dependencies are unavailable."""


@dataclass
class HighsSolverResult:
    solver_name: str
    run_status: str
    model_status: str
    objective_value: Optional[float]
    variables: Dict[str, float]
    reduced_costs: Dict[str, float]
    row_duals: Dict[str, float]
    summary: str

    def to_payload(self) -> dict:
        return {
            "solver_name": self.solver_name,
            "run_status": self.run_status,
            "model_status": self.model_status,
            "objective_value": self.objective_value,
            "variables": self.variables,
            "reduced_costs": self.reduced_costs,
            "row_duals": self.row_duals,
            "summary": self.summary,
        }


class HighsLPSolver:
    """Solve LP/MILP models supplied as .lp text via HiGHS."""

    def solve(self, lp_code: str, *, time_limit: Optional[float] = None) -> dict:
        if Highs is None or HighsStatus is None:
            raise MissingDependencyError(
                "HiGHS python bindings (highspy) are not installed. Install 'highspy' to use solve_lp."
            )
        lp_blob = (lp_code or "").strip()
        if not lp_blob:
            raise ValueError("lp_code must be a non-empty LP formatted string.")

        with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False) as tmp:
            tmp.write(lp_blob)
            tmp_path = tmp.name

        try:
            highs = Highs()
            if time_limit is not None:
                highs.setOptionValue("time_limit", float(time_limit))
            # Silence console logging so MCP stdio stays JSON-only.
            highs.setOptionValue("output_flag", False)
            highs.setOptionValue("log_to_console", False)
            status = highs.readModel(tmp_path)
            if status != HighsStatus.kOk:
                raise RuntimeError(f"HiGHS could not parse LP input (status={status}).")
            run_status = highs.run()
            if run_status != HighsStatus.kOk:
                raise RuntimeError(f"HiGHS solve failed (status={run_status}).")
            model_status = highs.getModelStatus()
            info = highs.getInfo()
            solution = highs.getSolution()
            lp = highs.getLp()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                LOGGER.warning("Failed to remove temporary LP file at %s", tmp_path)

        objective_value = getattr(info, "objective_function_value", None)
        col_values = list(getattr(solution, "col_value", []) or [])
        col_duals = list(getattr(solution, "col_dual", []) or [])
        row_duals = list(getattr(solution, "row_dual", []) or [])
        raw_col_names = list(getattr(lp, "col_names", []) or [])
        raw_row_names = list(getattr(lp, "row_names", []) or [])

        col_names = [_normalize_name(name) for name in raw_col_names]
        row_names = [_normalize_name(name) for name in raw_row_names]

        variables: Dict[str, float] = {}
        reduced_costs: Dict[str, float] = {}
        for idx, value in enumerate(col_values):
            fallback = col_names[idx] if idx < len(col_names) else None
            name = _resolve_name(
                _call_solver_name_getter(highs, "getColName", idx),
                fallback,
                f"x{idx}",
            )
            variables[name] = float(value)
            if idx < len(col_duals):
                reduced_costs[name] = float(col_duals[idx])
        row_dual_map: Dict[str, float] = {}
        for idx, value in enumerate(row_duals):
            fallback = row_names[idx] if idx < len(row_names) else None
            name = _resolve_name(
                _call_solver_name_getter(highs, "getRowName", idx),
                fallback,
                f"row{idx}",
            )
            row_dual_map[name] = float(value)

        run_status_str = getattr(run_status, "name", str(run_status))
        model_status_str = getattr(model_status, "name", str(model_status))
        summary_lines = [
            f"HiGHS run_status={run_status_str}, model_status={model_status_str}.",
        ]
        if objective_value is not None:
            summary_lines.append(f"Objective value = {objective_value}")
        if variables:
            summary_lines.append("Variables (first 10 shown):")
            for name, value in list(variables.items())[:10]:
                summary_lines.append(f"  {name} = {value}")
        summary = "\n".join(summary_lines)

        return HighsSolverResult(
            solver_name="highs",
            run_status=run_status_str,
            model_status=model_status_str,
            objective_value=objective_value,
            variables=variables,
            reduced_costs=reduced_costs,
            row_duals=row_dual_map,
            summary=summary,
        ).to_payload()


def _normalize_name(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, tuple) and raw:
        raw = raw[-1]
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw.decode("latin-1", errors="ignore")
    text = str(raw).strip()
    return text or None


def _call_solver_name_getter(highs_obj: Any, method_name: str, index: int) -> Optional[str]:
    getter = getattr(highs_obj, method_name, None)
    if getter is None:
        return None
    try:
        raw = getter(index)
    except Exception:  # pragma: no cover - defensive against exotic bindings
        return None
    return _normalize_name(raw)


def _resolve_name(primary: Optional[str], secondary: Optional[str], default: str) -> str:
    return primary or secondary or default


__all__ = [
    "HighsLPSolver",
    "HighsSolverResult",
    "MissingDependencyError",
    "_call_solver_name_getter",
    "_normalize_name",
    "_resolve_name",
]
