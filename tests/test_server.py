from __future__ import annotations

import asyncio

import pytest

from shared.lp_validator import validate_lp_text
from optimise_mcp.server import OptimiseMCPHandler, build_fastmcp_server


def test_fastmcp_server_exposes_solve_lp_and_validate_lp() -> None:
    handler = OptimiseMCPHandler()
    server = build_fastmcp_server(handler)
    tools = asyncio.run(server.get_tools())
    assert sorted(tools.keys()) == ["solve_lp", "validate_lp"]


def test_solve_lp_requires_lp_code() -> None:
    handler = OptimiseMCPHandler()
    with pytest.raises(ValueError):
        handler.solve_lp("")


def test_solve_lp_returns_structured_payload() -> None:
    calls: list[tuple[str, float | None]] = []

    class DummySolver:
        def solve(self, lp_code: str, time_limit: float | None = None) -> dict:
            calls.append((lp_code, time_limit))
            return {
                "summary": "ok",
                "model_status": "kOptimal",
                "run_status": "kOk",
            }

    handler = OptimiseMCPHandler(solver_factory=DummySolver)

    result = handler.solve_lp("Minimize obj: 0\nEnd")

    assert calls, "solve_lp should call into the solver"
    assert result.structured_content["model_status"] == "kOptimal"
    assert "ok" in getattr(result.content[0], "text", "")


def test_solve_lp_reports_quota_pressure() -> None:
    handler = OptimiseMCPHandler(max_parallel_solvers=1, solver_factory=lambda: None)  # type: ignore[arg-type]

    acquired = handler._solver_quota._semaphore.acquire(blocking=False)  # type: ignore[attr-defined]
    assert acquired
    try:
        result = handler.solve_lp("Minimize obj: 0\nEnd")
    finally:
        handler._solver_quota._semaphore.release()  # type: ignore[attr-defined]

    assert result.structured_content["error"]
    assert "temporarily unavailable" in getattr(result.content[0], "text", "")


def test_validate_lp_flags_nonlinear_ops() -> None:
    issues = validate_lp_text(
        """
Subject To
 bad: wheat * corn <= 10
End
""".strip()
    )

    assert issues
    assert issues[0].line_number == 2
    assert "Non-linear" in issues[0].message


def test_validate_lp_flags_variable_on_rhs() -> None:
    issues = validate_lp_text(
        """
Subject To
 bad: wheat + corn <= steel
End
""".strip()
    )

    assert issues
    assert any("right-hand side" in issue.message.lower() for issue in issues)


def test_validate_lp_allows_quadratic_objective_block() -> None:
    issues = validate_lp_text(
        """
Minimize
 obj: x + y + [ x^2 + 4 x * y + 7 y^2 ]/2
Subject To
 c1: x + y >= 0
End
""".strip()
    )

    assert issues == []


def test_validate_lp_rejects_quadratic_block_without_half() -> None:
    issues = validate_lp_text(
        """
Minimize
 obj: x + y + [ x^2 + y^2 ]
Subject To
 c1: x + y >= 0
End
""".strip()
    )

    assert issues
    assert any("/2" in issue.message for issue in issues)


def test_solve_lp_short_circuits_when_lp_invalid() -> None:
    def _failing_solver():  # type: ignore[return-value]
        raise AssertionError("solver should not be invoked when validation fails")

    handler = OptimiseMCPHandler(solver_factory=_failing_solver)
    result = handler.solve_lp("Subject To\n bad: x * y <= 1\nEnd")

    assert result.structured_content["error"] == "LP validation failed"
    assert result.structured_content["issues"]


def test_validate_lp_tool_detects_issues() -> None:
    handler = OptimiseMCPHandler()
    result = handler.validate_lp("Subject To\n bad: x * y <= 1\nEnd")

    assert result.structured_content["valid"] is False
    assert result.structured_content["issues"]


def test_validate_lp_tool_accepts_valid_lp() -> None:
    handler = OptimiseMCPHandler()
    result = handler.validate_lp("Minimize\n obj: x + y\nSubject To\n c1: x >= 0\nEnd")

    assert result.structured_content["valid"] is True
    assert result.structured_content["issues"] == []
