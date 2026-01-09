"""Tests for LangChain tools."""

from __future__ import annotations

import pytest

from langchain_optimise.tools import create_validate_lp_tool, create_solve_lp_tool


def test_create_validate_lp_tool() -> None:
    tool = create_validate_lp_tool()
    assert tool.name == "validate_lp"


def test_validate_lp_tool_accepts_valid_lp() -> None:
    tool = create_validate_lp_tool()
    result = tool.invoke({"lp_code": "Minimize\n obj: x + y\nSubject To\n c1: x >= 0\nEnd"})

    assert result["valid"] is True
    assert result["issues"] == []


def test_validate_lp_tool_detects_issues() -> None:
    tool = create_validate_lp_tool()
    result = tool.invoke({"lp_code": "Subject To\n bad: x * y <= 1\nEnd"})

    assert result["valid"] is False
    assert len(result["issues"]) > 0


def test_validate_lp_tool_rejects_empty_code() -> None:
    tool = create_validate_lp_tool()
    result = tool.invoke({"lp_code": ""})

    assert result["valid"] is False
    assert result["issues"][0]["message"] == "lp_code is required"


def test_create_solve_lp_tool() -> None:
    tool = create_solve_lp_tool()
    assert tool.name == "solve_lp"


def test_solve_lp_tool_rejects_invalid_lp() -> None:
    tool = create_solve_lp_tool()
    result = tool.invoke({"lp_code": "Subject To\n bad: x * y <= 1\nEnd"})

    assert "error" in result
    assert result["error"] == "LP validation failed"
    assert len(result["issues"]) > 0


def test_solve_lp_tool_rejects_empty_code() -> None:
    tool = create_solve_lp_tool()
    result = tool.invoke({"lp_code": ""})

    assert result["error"] == "lp_code is required"
