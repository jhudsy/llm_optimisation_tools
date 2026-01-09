"""
Multi-agent LangChain workflow with Modeller and Checker agents.

This is a CLI wrapper for the modeller-checker workflow.
For MCP integration or LangChain tools, see:
- src/modeller_checker/mcp.py (MCP server)
- src/langchain_optimise/modeller_checker_tool.py (LangChain tool)
"""

import asyncio
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from modeller_checker.config import load_config, create_llms_from_config
from modeller_checker.workflow import run_modeller_checker_workflow
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool,
)


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-agent LangChain workflow with Modeller and Checker"
    )
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        default=(
            "We have 110 acres of land. We can plant wheat or corn. "
            "Wheat yields $40 profit per acre and requires 3 labour hours per acre. "
            "Corn yields $30 profit per acre and requires 2 labour hours per acre. "
            "We have 240 labour hours available. "
            "How should we allocate the land to maximize profit?"
        ),
        help="Problem description",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=5,
        help="Max modeller-checker iterations (default: 5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml (default: repo_root/config.yaml)",
    )
    # Legacy args for backwards compatibility
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="(Deprecated) Ollama model to use. Use config.yaml instead.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        help="(Deprecated) LLM temperature. Use config.yaml instead.",
    )
    
    args = parser.parse_args()
    
    if args.model or args.temperature:
        print(
            "Warning: --model and --temperature are deprecated. "
            "Please use config.yaml to configure LLMs.",
            file=sys.stderr,
        )
    
    # Load config and create LLMs
    modeller_llm, checker_llm = create_llms_from_config(args.config)
    
    # Create tools
    validate_tool = create_validate_minizinc_tool()
    solve_tool = create_solve_minizinc_tool()
    
    # Run workflow
    results = await run_modeller_checker_workflow(
        problem=args.problem,
        modeller_llm=modeller_llm,
        checker_llm=checker_llm,
        validate_tool=validate_tool,
        solve_tool=solve_tool,
        max_iterations=args.iterations,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Iterations: {results['iterations']}")
    print(f"Checker Approved: {results['checker_approval']}")
    print(f"Success: {results['success']}")
    print(f"\nFinal Response:\n{results['final_response']}")
    
    if results['mzn_code']:
        print(f"\nFinal MiniZinc Code:\n{results['mzn_code']}")


if __name__ == "__main__":
    asyncio.run(main())
