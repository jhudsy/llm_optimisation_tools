"""
LangChain tool factory for modeller-checker workflow.

Provides a LangChain tool that wraps the dual-agent workflow for direct use
in LangChain agents without MCP overhead.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from modeller_checker.config import load_config, create_llms_from_config
from modeller_checker.workflow import run_modeller_checker_workflow
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool,
)


class ModellerCheckerInput(BaseModel):
    """Input schema for modeller-checker workflow tool."""
    
    problem: str = Field(
        description=(
            "Natural language problem description. "
            "Should include: decision variables, constraints, objective function, "
            "and any resource limits or requirements."
        )
    )
    max_iterations: int = Field(
        default=5,
        description="Maximum number of modeller-checker refinement iterations"
    )


class ModellerCheckerTool(BaseTool):
    """LangChain tool for modeller-checker workflow."""
    
    name: str = "modeller_checker_workflow"
    description: str = (
        "Dual-agent workflow for optimization problem modeling. "
        "A Modeller agent creates MiniZinc models from problem descriptions, "
        "a Checker agent validates correctness, and feedback loops refine the model. "
        "Once approved, the model is solved and the optimal solution is returned. "
        "Use this for complex optimization problems requiring validation."
    )
    args_schema: type[BaseModel] = ModellerCheckerInput
    
    modeller_llm: Optional[object] = None
    checker_llm: Optional[object] = None
    validate_tool: Optional[object] = None
    solve_tool: Optional[object] = None
    verbose: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, problem: str, max_iterations: int = 5) -> str:
        """Synchronous wrapper for async workflow."""
        return asyncio.run(self._arun(problem, max_iterations))
    
    async def _arun(self, problem: str, max_iterations: int = 5) -> str:
        """Run the modeller-checker workflow asynchronously."""
        result = await run_modeller_checker_workflow(
            problem=problem,
            modeller_llm=self.modeller_llm,
            checker_llm=self.checker_llm,
            validate_tool=self.validate_tool,
            solve_tool=self.solve_tool,
            max_iterations=max_iterations,
            verbose=self.verbose,
        )
        
        response = f"""Modeller-Checker Workflow Results:

Success: {result['success']}
Checker Approval: {result['checker_approval']}
Iterations: {result['iterations']}

Final Response:
{result['final_response']}
"""
        
        if result['mzn_code']:
            response += f"\nFinal MiniZinc Model:\n{result['mzn_code']}"
        
        return response


def create_modeller_checker_tool(
    config_path: Optional[str] = None,
    verbose: bool = False,
) -> ModellerCheckerTool:
    """
    Create a LangChain tool for the modeller-checker workflow.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
        verbose: Print detailed workflow steps
    
    Returns:
        Configured ModellerCheckerTool instance
    
    Example:
        >>> from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool
        >>> tool = create_modeller_checker_tool(verbose=True)
        >>> result = tool.invoke({
        ...     "problem": "We have 110 acres of land. We can plant wheat or corn...",
        ...     "max_iterations": 5
        ... })
    """
    # Load config and create LLMs
    modeller_llm, checker_llm = create_llms_from_config(config_path)
    
    # Create validation and solver tools
    validate_tool = create_validate_minizinc_tool()
    solve_tool = create_solve_minizinc_tool()
    
    return ModellerCheckerTool(
        modeller_llm=modeller_llm,
        checker_llm=checker_llm,
        validate_tool=validate_tool,
        solve_tool=solve_tool,
        verbose=verbose,
    )
