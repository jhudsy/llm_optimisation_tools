"""
LangChain tool for complex 5-agent workflow.

Provides a LangChain tool that wraps the complex workflow for direct use
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

from modeller_checker.config import load_config, create_complex_workflow_llms
from modeller_checker.complex_workflow import run_complex_workflow, format_formulation_for_display
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool,
)


class ComplexWorkflowInput(BaseModel):
    """Input schema for complex workflow tool."""
    
    problem: str = Field(
        description=(
            "Natural language problem description. "
            "Should include: decision variables, constraints, objective function, "
            "and any resource limits or requirements."
        )
    )
    max_iterations: int = Field(
        default=None,
        description="Maximum number of workflow iterations (uses config default if not specified)"
    )


class ComplexWorkflowTool(BaseTool):
    """LangChain tool for complex 5-agent workflow."""
    
    name: str = "complex_optimization_workflow"
    description: str = (
        "Complex 5-agent workflow for optimization problem modeling. "
        "Uses specialized agents: "
        "1) Formulator creates mathematical equations from problem description, "
        "2) Equation Checker validates equations match the problem, "
        "3) Translator converts equations to MiniZinc code, "
        "4) Code Checker validates MiniZinc implementation, "
        "5) Solver Executor runs solver and diagnoses errors, routing feedback appropriately. "
        "Returns optimal solution with complete formulation and code. "
        "Use for complex optimization requiring validated mathematical formulation."
    )
    args_schema: type[BaseModel] = ComplexWorkflowInput
    
    formulator_llm: Optional[object] = None
    equation_checker_llm: Optional[object] = None
    translator_llm: Optional[object] = None
    code_checker_llm: Optional[object] = None
    solver_executor_llm: Optional[object] = None
    validate_tool: Optional[object] = None
    solve_tool: Optional[object] = None
    verbose: bool = False
    default_max_iterations: int = 10
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, problem: str, max_iterations: int = None) -> str:
        """Synchronous wrapper for async workflow."""
        return asyncio.run(self._arun(problem, max_iterations))
    
    async def _arun(self, problem: str, max_iterations: int = None) -> str:
        """Run the complex workflow asynchronously."""
        # Use provided max_iterations or fall back to default from config
        iterations = max_iterations if max_iterations is not None else self.default_max_iterations
        
        result = await run_complex_workflow(
            problem=problem,
            formulator_llm=self.formulator_llm,
            equation_checker_llm=self.equation_checker_llm,
            translator_llm=self.translator_llm,
            code_checker_llm=self.code_checker_llm,
            solver_executor_llm=self.solver_executor_llm,
            validate_tool=self.validate_tool,
            solve_tool=self.solve_tool,
            max_iterations=iterations,
            verbose=self.verbose,
        )
        
        response = f"""Complex Workflow Results:

Success: {result['success']}
Iterations: {result['iterations']}
Workflow Trace: {' -> '.join(result['workflow_trace'])}

Final Response:
{result['final_response']}
"""
        
        if result.get('formulation'):
            response += f"\n{'='*60}\nMATHEMATICAL FORMULATION\n{'='*60}\n"
            response += format_formulation_for_display(result['formulation'])
        
        if result.get('mzn_code'):
            response += f"\n\n{'='*60}\nMINIZINC MODEL\n{'='*60}\n{result['mzn_code']}"
        
        if result.get('solution'):
            response += f"\n\n{'='*60}\nSOLUTION\n{'='*60}\n"
            for var, value in result['solution'].items():
                response += f"  {var} = {value}\n"
        
        return response


def create_complex_workflow_tool(
    config_path: Optional[str] = None,
    verbose: bool = False,
) -> ComplexWorkflowTool:
    """
    Create a LangChain tool for the complex 5-agent workflow.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
        verbose: Print detailed workflow steps
    
    Returns:
        Configured ComplexWorkflowTool instance
    
    Example:
        >>> from langchain_optimise.complex_workflow_tool import create_complex_workflow_tool
        >>> tool = create_complex_workflow_tool(verbose=True)
        >>> result = tool.invoke({
        ...     "problem": "We have 110 acres of land. We can plant wheat or corn...",
        ...     "max_iterations": 10
        ... })
    """
    # Load config and create LLMs
    config = load_config(config_path)
    (formulator_llm, equation_checker_llm, translator_llm, 
     code_checker_llm, solver_executor_llm) = create_complex_workflow_llms(config_path)
    
    # Get workflow config
    workflow_config = config.get("workflow", {})
    default_max_iterations = workflow_config.get("max_iterations", 10)
    solver_backend = workflow_config.get("solver_backend", "coinbc")
    # Map 'mzn' to 'coinbc' for backwards compatibility
    if solver_backend == "mzn":
        solver_backend = "coinbc"
    
    # Create validation and solver tools
    from mzn.solver import MiniZincSolver
    validate_tool = create_validate_minizinc_tool()
    solve_tool = create_solve_minizinc_tool(
        solver_factory=lambda: MiniZincSolver(solver_backend=solver_backend)
    )
    
    return ComplexWorkflowTool(
        formulator_llm=formulator_llm,
        equation_checker_llm=equation_checker_llm,
        translator_llm=translator_llm,
        code_checker_llm=code_checker_llm,
        solver_executor_llm=solver_executor_llm,
        validate_tool=validate_tool,
        solve_tool=solve_tool,
        verbose=verbose,
        default_max_iterations=default_max_iterations,
    )
