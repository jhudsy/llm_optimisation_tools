"""Lightweight validation for MiniZinc model text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re


@dataclass
class MiniZincValidationIssue:
    """Represents a structural problem detected in a MiniZinc model."""

    line_number: int
    message: str
    line_content: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "line_number": self.line_number,
            "message": self.message,
            "line_content": self.line_content.rstrip("\n"),
        }


def validate_minizinc_text(mzn_code: str) -> List[MiniZincValidationIssue]:
    """Scan user-provided MiniZinc text and return structural issues, if any.
    
    Basic validation for:
    - Matching brackets and parentheses
    - Required keywords (at least one var, constraint, or solve)
    """
    issues: list[MiniZincValidationIssue] = []

    lines = mzn_code.splitlines()

    # Check for matching brackets
    bracket_stack: list[tuple[str, int]] = []
    bracket_pairs = {"(": ")", "[": "]", "{": "}"}

    for idx, raw_line in enumerate(lines, start=1):
        # Skip comments
        line = raw_line.split("%")[0] if "%" in raw_line else raw_line

        for char_idx, char in enumerate(line):
            if char in bracket_pairs:
                bracket_stack.append((char, idx))
            elif char in bracket_pairs.values():
                if not bracket_stack:
                    issues.append(
                        MiniZincValidationIssue(
                            line_number=idx,
                            message=f"Unmatched closing bracket '{char}' without opening counterpart.",
                            line_content=raw_line,
                        )
                    )
                else:
                    opening, open_line = bracket_stack[-1]
                    expected_closing = bracket_pairs[opening]
                    if char != expected_closing:
                        issues.append(
                            MiniZincValidationIssue(
                                line_number=idx,
                                message=(
                                    f"Mismatched brackets: '{opening}' (line {open_line}) "
                                    f"closed by '{char}' instead of '{expected_closing}'."
                                ),
                                line_content=raw_line,
                            )
                        )
                    bracket_stack.pop()

    # Check for unclosed brackets
    for bracket, line_no in bracket_stack:
        expected_closing = bracket_pairs[bracket]
        issues.append(
            MiniZincValidationIssue(
                line_number=line_no,
                message=(
                    f"Unclosed bracket '{bracket}'; expected '{expected_closing}' "
                    f"before end of model."
                ),
                line_content=lines[line_no - 1] if line_no <= len(lines) else "",
            )
        )

    # Check for comma-separated variable declarations (common error)
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.split("%")[0].strip()
        # Detect patterns like: var float: x, y; or var 0..10: a, b;
        if re.search(r"\bvar\s+[^;]*,\s*\w+\s*;", line):
            issues.append(
                MiniZincValidationIssue(
                    line_number=idx,
                    message=(
                        "Each variable must be declared on its own line. "
                        "Split comma-separated declarations. "
                        "Example: 'var float: x;' then 'var float: y;' on separate lines."
                    ),
                    line_content=raw_line,
                )
            )

    # Check for malformed output statements (arrays of vars)
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.split("%")[0].strip()
        if "output" in line.lower():
            # Check for incorrect patterns like: output [x, y, z];
            if re.search(r"output\s*\[\s*\w+\s*(,\s*\w+\s*)+\]", line, re.IGNORECASE):
                issues.append(
                    MiniZincValidationIssue(
                        line_number=idx,
                        message=(
                            "Output statement must use show() for variables and mix with strings. "
                            "Example: output [\"x=\", show(x), \" y=\", show(y)];"
                        ),
                        line_content=raw_line,
                    )
                )

    # Check for double type declarations like: var 0..110: int: wheat;
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.split("%")[0].strip()
        if re.search(r"\bvar\s+[^:]+:\s*(int|float|bool)\s*:", line):
            issues.append(
                MiniZincValidationIssue(
                    line_number=idx,
                    message=(
                        "Remove redundant type annotation. Use 'var domain: name;' not 'var domain: type: name;'. "
                        "Example: 'var 0..110: wheat;' not 'var 0..110: int: wheat;'"
                    ),
                    line_content=raw_line,
                )
            )

    # Check for minimum model structure
    code_lower = mzn_code.lower()
    has_var = re.search(r"\bvar\s+", code_lower)
    has_constraint = re.search(r"\bconstraint\b", code_lower)
    has_solve = re.search(r"\bsolve\b", code_lower)

    if not (has_var or has_constraint or has_solve):
        issues.append(
            MiniZincValidationIssue(
                line_number=1,
                message=(
                    "MiniZinc model must contain at least one of: variable declaration (var), "
                    "constraint, or solve statement."
                ),
                line_content=lines[0] if lines else "",
            )
        )

    return issues


__all__ = ["MiniZincValidationIssue", "validate_minizinc_text"]
