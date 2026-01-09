#!/usr/bin/env python3
"""Backward-compatible wrapper that always runs the HTTP transport."""

from __future__ import annotations

from mcp_minizinc_roundtrip import main as shared_main


if __name__ == "__main__":
    raise SystemExit(shared_main(default_transport="http", force_transport="http"))
