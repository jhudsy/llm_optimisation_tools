from __future__ import annotations

from shared.solver import _call_solver_name_getter, _normalize_name, _resolve_name


def test_normalize_name_handles_bytes_and_whitespace() -> None:
    assert _normalize_name(b" wheat ") == "wheat"
    assert _normalize_name("  ") is None
    assert _normalize_name(("unused", "corn")) == "corn"
    assert _normalize_name(None) is None


def test_call_solver_name_getter_invokes_highs_method() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def getColName(self, index: int) -> str:  # noqa: N802 - mirrors HiGHS API
            self.calls.append(index)
            return f"x{index}"

    dummy = Dummy()
    name = _call_solver_name_getter(dummy, "getColName", 3)

    assert name == "x3"
    assert dummy.calls == [3]


def test_resolve_name_prefers_primary_then_secondary_then_default() -> None:
    assert _resolve_name("primary", "secondary", "default") == "primary"
    assert _resolve_name(None, "secondary", "default") == "secondary"
    assert _resolve_name(None, None, "default") == "default"
