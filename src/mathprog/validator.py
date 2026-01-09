from __future__ import annotations

"""Lightweight validation for LP (.lp) text prior to handing it to HiGHS."""

from dataclasses import dataclass
import re
from typing import Iterator, List

_LINE_PATTERN = re.compile(r"(<=|>=|=)")
_NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_SECTION_PREFIXES = {
    "maximize": "objective",
    "maximise": "objective",
    "minimize": "objective",
    "minimise": "objective",
    "subject to": "constraints",
    "such that": "constraints",
    "bounds": "bounds",
    "binary": "binary",
    "general": "general",
    "generals": "general",
    "integer": "general",
    "integers": "general",
    "end": "end",
}
_ALLOWED_RHS_TOKENS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


@dataclass
class LPValidationIssue:
    """Represents a structural problem detected in an LP body."""

    line_number: int
    message: str
    line_content: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "line_number": self.line_number,
            "message": self.message,
            "line_content": self.line_content.rstrip("\n"),
        }


def validate_lp_text(lp_code: str) -> List[LPValidationIssue]:
    """Scan user-provided LP text and return structural issues, if any."""

    issues: list[LPValidationIssue] = []
    section: str | None = None
    objective_state = _ObjectiveQuadraticState()

    for idx, raw_line in enumerate(lp_code.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        section, remainder = _strip_section_prefix(stripped, section)
        if not remainder:
            continue

        line_for_checks = remainder
        if section == "objective":
            line_for_checks, quad_issues = _sanitize_objective_line(
                remainder, raw_line, idx, objective_state
            )
            issues.extend(quad_issues)
        else:
            quad_issue = _detect_quadratic_bracket_outside_objective(remainder)
            if quad_issue:
                issues.append(
                    LPValidationIssue(
                        line_number=idx,
                        message=quad_issue,
                        line_content=raw_line,
                    )
                )

        # Reject obvious non-linear operators.
        if not line_for_checks:
            continue

        nonlinear_char = _find_nonlinear_operator(line_for_checks)
        if nonlinear_char:
            issues.append(
                LPValidationIssue(
                    line_number=idx,
                    message=(
                        f"Non-linear operator '{nonlinear_char}' detected; specify numeric"
                        " coefficients instead of using products/division/exponents."
                    ),
                    line_content=raw_line,
                )
            )
            continue

        if section == "constraints":
            rhs_issue = _detect_rhs_variable(remainder)
            if rhs_issue:
                issues.append(
                    LPValidationIssue(
                        line_number=idx,
                        message=rhs_issue,
                        line_content=raw_line,
                    )
                )

    if objective_state.quadratic_open:
        issues.append(
            LPValidationIssue(
                line_number=objective_state.open_line_number or 0,
                message="Quadratic objective block was opened but not closed before leaving the objective section.",
                line_content=objective_state.open_line_content or "",
            )
        )

    return issues


def _strip_section_prefix(line: str, current_section: str | None) -> tuple[str | None, str]:
    lowered = line.lower()
    for prefix, section in _SECTION_PREFIXES.items():
        if lowered.startswith(prefix):
            remainder = line[len(prefix) :].lstrip()
            return section, remainder
    return current_section, line


def _find_nonlinear_operator(line: str) -> str | None:
    for char in ("*", "/", "^"):
        if char in line:
            return char
    return None


def _detect_rhs_variable(line: str) -> str | None:
    matches = list(_LINE_PATTERN.finditer(line))
    if not matches:
        return None
    last = matches[-1]
    rhs = line[last.end() :].strip()
    if not rhs:
        return "Constraint is missing a numeric constant on the right-hand side."

    scrubbed = _remove_numeric_literals(rhs)
    scrubbed_lower = scrubbed.lower()
    for token in _ALLOWED_RHS_TOKENS:
        scrubbed_lower = scrubbed_lower.replace(token, " ")
    if re.search(r"[A-Za-z_]", scrubbed_lower):
        return "Decision variables should not appear on the right-hand side of a constraint."
    return None


def _remove_numeric_literals(text: str) -> str:
    return _NUMBER_PATTERN.sub(" ", text)


@dataclass
class _ObjectiveQuadraticState:
    quadratic_open: bool = False
    quadratic_seen: bool = False
    open_line_number: int | None = None
    open_line_content: str | None = None


def _sanitize_objective_line(
    remainder: str,
    raw_line: str,
    line_number: int,
    objective_state: _ObjectiveQuadraticState,
) -> tuple[str, list[LPValidationIssue]]:
    issues: list[LPValidationIssue] = []
    sanitized_chars: list[str] = []
    idx = 0
    length = len(remainder)

    while idx < length:
        char = remainder[idx]
        if char == "[":
            if objective_state.quadratic_open:
                issues.append(
                    LPValidationIssue(
                        line_number=line_number,
                        message="Nested quadratic blocks are not allowed in the objective.",
                        line_content=raw_line,
                    )
                )
            elif objective_state.quadratic_seen:
                issues.append(
                    LPValidationIssue(
                        line_number=line_number,
                        message="Only one quadratic [ ... ]/2 block may appear in the objective.",
                        line_content=raw_line,
                    )
                )
            objective_state.quadratic_open = True
            objective_state.quadratic_seen = True
            objective_state.open_line_number = line_number
            objective_state.open_line_content = raw_line
            idx += 1
            continue
        if char == "]":
            if not objective_state.quadratic_open:
                issues.append(
                    LPValidationIssue(
                        line_number=line_number,
                        message="] encountered in the objective without a matching [.",
                        line_content=raw_line,
                    )
                )
                idx += 1
                continue
            objective_state.quadratic_open = False
            idx += 1
            idx = _skip_whitespace(remainder, idx)
            if idx >= length or remainder[idx] != "/":
                issues.append(
                    LPValidationIssue(
                        line_number=line_number,
                        message="Quadratic objective block must end with '/2'.",
                        line_content=raw_line,
                    )
                )
                idx = length
                continue
            idx += 1
            idx = _skip_whitespace(remainder, idx)
            if idx >= length or remainder[idx] != "2":
                issues.append(
                    LPValidationIssue(
                        line_number=line_number,
                        message="Quadratic objective block must end with '/2'.",
                        line_content=raw_line,
                    )
                )
                idx = length
                continue
            idx += 1
            continue
        if objective_state.quadratic_open:
            idx += 1
            continue
        sanitized_chars.append(char)
        idx += 1

    return "".join(sanitized_chars).strip(), issues


def _detect_quadratic_bracket_outside_objective(line: str) -> str | None:
    if "[" in line or "]" in line:
        return "Quadratic [ ... ] blocks are only allowed in the objective section."
    return None


def _skip_whitespace(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


__all__ = ["LPValidationIssue", "validate_lp_text"]
