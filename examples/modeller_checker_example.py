"""
Example: Using the modeller-checker workflow via LangChain tool.

This demonstrates direct Python usage without MCP overhead.
"""

import asyncio
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool


async def main():
    """Run modeller-checker workflow example."""
    # Create tool (loads config.yaml from repo root)
    print("Creating modeller-checker tool...")
    tool = create_modeller_checker_tool(verbose=True)
    
    # Define problem
    problem = """
    We have 110 acres of land. We can plant wheat or corn.
    Wheat yields $40 profit per acre and requires 3 labour hours per acre.
    Corn yields $30 profit per acre and requires 2 labour hours per acre.
    We have 240 labour hours available.
    How should we allocate the land to maximize profit?
    """
    
    print("\nRunning workflow...")
    print("=" * 80)
    
    # Run workflow
    result = await tool._arun(problem=problem, max_iterations=5)
    
    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
