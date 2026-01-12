"""
Test script for the optimization workflow.

Run: python scripts/workflow_test.py
"""

import asyncio
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from modeller_checker.config import load_config, create_llms_from_config
from modeller_checker.workflow import run_workflow, format_formulation_for_display
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool,
)
from mzn.solver import MiniZincSolver


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test optimization workflow"
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
        default=None,
        help="Max workflow iterations (default from config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml (default: repo_root/config.yaml)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    workflow_config = config.get("workflow", {})
    max_iterations = args.iterations if args.iterations is not None else workflow_config.get("max_iterations", 10)
    
    # Create LLMs for all 5 agents
    print("Loading LLMs for workflow...")
    (formulator_llm, equation_checker_llm, translator_llm, 
     code_checker_llm, solver_executor_llm) = create_llms_from_config(args.config)
    
    # Create tools
    print("Initializing tools...")
    validate_tool = create_validate_minizinc_tool()
    
    solver_backend = workflow_config.get("solver_backend", "coinbc")
    if solver_backend == "mzn":
        solver_backend = "coinbc"
    
    solve_tool = create_solve_minizinc_tool(
        solver_factory=lambda: MiniZincSolver(solver_backend=solver_backend)
    )
    
    # Wrap solve_tool for async
    class SyncToolWrapper:
        """Wraps sync tool to work in async context."""
        def __init__(self, tool):
            self.tool = tool
        
        async def invoke_async(self, input_dict):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.tool.invoke(input_dict))
    
    async_solve_tool = SyncToolWrapper(solve_tool)
    
    print("\n" + "=" * 80)
    print("Starting optimization workflow...")
    print("=" * 80)
    
    # Run workflow
    results = await run_workflow(
        problem=args.problem,
        formulator_llm=formulator_llm,
        equation_checker_llm=equation_checker_llm,
        translator_llm=translator_llm,
        code_checker_llm=code_checker_llm,
        solver_executor_llm=solver_executor_llm,
        validate_tool=validate_tool,
        solve_tool=async_solve_tool,
        max_iterations=max_iterations,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Iterations: {results['iterations']}")
    print(f"Success: {results['success']}")
    print(f"\nWorkflow Trace: {' -> '.join(results['workflow_trace'])}")
    print(f"\nFinal Response:\n{results['final_response']}")
    
    if results.get('formulation'):
        print(f"\n{'='*80}")
        print("MATHEMATICAL FORMULATION")
        print("=" * 80)
        print(format_formulation_for_display(results['formulation']))
    
    if results.get('mzn_code'):
        print(f"\n{'='*80}")
        print("MINIZINC CODE")
        print("=" * 80)
        print(results['mzn_code'])
    
    if results.get('solution'):
        print(f"\n{'='*80}")
        print("SOLUTION")
        print("=" * 80)
        for var, value in results['solution'].items():
            print(f"  {var} = {value}")


if __name__ == "__main__":
    asyncio.run(main())
