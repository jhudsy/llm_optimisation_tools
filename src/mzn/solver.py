"""MiniZinc constraint programming solver with pluggable backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

# Avoid local package name collision by loading the external 'minizinc' package
# from site-packages under a distinct alias.
import os
import sys
import importlib.util
import sysconfig
from types import ModuleType

def _load_external_minizinc() -> ModuleType | None:
    try:
        # Compute site-packages paths
        purelib = sysconfig.get_paths().get("purelib")
        platlib = sysconfig.get_paths().get("platlib")
        site_paths = [p for p in (purelib, platlib) if p]
        if not site_paths:
            return None
        # Temporarily prioritise site-packages to avoid local package shadowing
        original_sys_path = list(sys.path)
        # Remove the project src path if present at the front
        prj_src = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if prj_src in sys.path:
            sys.path.remove(prj_src)
        # Prepend site-packages
        for sp in reversed(site_paths):
            if sp not in sys.path:
                sys.path.insert(0, sp)
        try:
            import importlib
            ext = importlib.import_module("minizinc")
            # Basic sanity: ensure expected attributes exist
            if hasattr(ext, "Solver") and hasattr(ext, "Model") and hasattr(ext, "Instance"):
                return ext
        finally:
            # Restore sys.path
            sys.path[:] = original_sys_path
    except Exception:
        return None
    return None

minizinc_external = _load_external_minizinc()


LOGGER = logging.getLogger("mzn.solver")


class MissingDependencyError(RuntimeError):
    """Raised when solver dependencies are unavailable."""


@dataclass
class MiniZincSolverResult:
    solver_name: str
    solver_backend: str
    run_status: str
    objective_value: Optional[float]
    variables: Dict[str, Any]
    summary: str

    def to_payload(self) -> dict:
        return {
            "solver_name": self.solver_name,
            "solver_backend": self.solver_backend,
            "run_status": self.run_status,
            "objective_value": self.objective_value,
            "variables": self.variables,
            "summary": self.summary,
        }


class MiniZincSolver:
    """Solve constraint programming models supplied as MiniZinc text."""

    def __init__(
        self,
        solver_backend: str = "coinbc",
        time_limit: Optional[float] = None,
    ) -> None:
        """Initialize MiniZinc solver.

        Args:
            solver_backend: Name of MiniZinc solver backend (default: 'coinbc')
            time_limit: Global time limit in seconds
        """
        self.solver_backend = solver_backend
        self.time_limit = time_limit
        # Detect available execution mode: Python bindings vs CLI
        self._use_python_api = minizinc_external is not None
        if not self._use_python_api:
            # Verify CLI availability
            import shutil
            if shutil.which("minizinc") is None:
                raise MissingDependencyError(
                    "MiniZinc CLI not found in PATH and Python bindings unavailable. Install MiniZinc (brew install minizinc) or ensure bindings are importable."
                )

    def solve(self, mzn_code: str, *, time_limit: Optional[float] = None) -> dict:
        """Solve a MiniZinc model.

        Args:
            mzn_code: MiniZinc model code as string
            time_limit: Optional override for solver time limit in seconds

        Returns:
            Dictionary with solver results and variable assignments
        """
        mzn_blob = (mzn_code or "").strip()
        if not mzn_blob:
            raise ValueError("mzn_code must be a non-empty MiniZinc model string.")

        # Determine effective time limit
        effective_time_limit = time_limit or self.time_limit

        try:
            objective_value = None
            variables: Dict[str, Any] = {}
            status_str = "Unknown"

            if self._use_python_api:
                # Python API path
                solver = minizinc_external.Solver.lookup(self.solver_backend)
                model = minizinc_external.Model()
                model.add_string(mzn_blob)

                instance = minizinc_external.Instance(solver, model)
                if effective_time_limit is not None:
                    instance.options["time_limit"] = int(effective_time_limit * 1000)  # ms

                result = instance.solve()

                if result.status == minizinc_external.Status.OPTIMAL_FOUND or result.status == minizinc_external.Status.SATISFIED:
                    if hasattr(result, "__getitem__") and hasattr(model, "items"):
                        for item in getattr(model, "items", []):
                            try:
                                var_name = getattr(item, "name", None)
                                if var_name:
                                    variables[var_name] = result[var_name]
                            except (KeyError, AttributeError):
                                pass
                    if hasattr(result, "objective") and result.objective is not None:
                        objective_value = float(result.objective)
                status_str = result.status.name if hasattr(result.status, "name") else str(result.status)
            else:
                # CLI fallback path
                import tempfile
                import subprocess
                import re

                with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False) as tf:
                    tf.write(mzn_blob)
                    tf.flush()
                    mzn_path = tf.name
                cmd = ["minizinc"]
                if self.solver_backend:
                    cmd += ["--solver", self.solver_backend]
                if effective_time_limit is not None:
                    cmd += ["-t", str(int(effective_time_limit * 1000))]  # ms
                cmd += [mzn_path]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()
                if proc.returncode != 0:
                    raise RuntimeError(f"MiniZinc CLI failed (code {proc.returncode}): {stderr or stdout}")
                # Parse simple name=value pairs from output
                for line in stdout.splitlines():
                    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\s*", line)
                    if m:
                        name, val = m.group(1), m.group(2)
                        variables[name] = val
                status_str = "success" if variables else "success"

            summary_lines = [f"MiniZinc solve completed with status: {status_str}"]
            if objective_value is not None:
                summary_lines.append(f"Objective value = {objective_value}")
            if variables:
                summary_lines.append("Variables (first 10 shown):")
                for name, value in list(variables.items())[:10]:
                    summary_lines.append(f"  {name} = {value}")

            return MiniZincSolverResult(
                solver_name="minizinc",
                solver_backend=self.solver_backend,
                run_status=status_str,
                objective_value=objective_value,
                variables=variables,
                summary="\n".join(summary_lines),
            ).to_payload()

        except Exception as exc:
            raise RuntimeError(f"MiniZinc solver failed: {exc}") from exc


__all__ = ["MiniZincSolver", "MiniZincSolverResult", "MissingDependencyError"]
