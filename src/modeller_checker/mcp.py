"""
MCP server for Modeller-Checker workflow.

Exposes the dual-agent optimization modeling workflow via MCP protocol.
Supports both stdio (for Ollama) and HTTP transports.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from fastmcp import FastMCP
from langchain_core.messages import BaseMessage

from modeller_checker.config import load_config, create_llms_from_config, create_complex_workflow_llms
from modeller_checker.workflow import run_modeller_checker_workflow
from modeller_checker.complex_workflow import run_complex_workflow
from mzn.solver import MiniZincSolver
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool,
    create_solve_minizinc_tool_async,
)


# Global configuration
_config = None
_modeller_llm = None
_checker_llm = None
_formulator_llm = None
_equation_checker_llm = None
_translator_llm = None
_code_checker_llm = None
_solver_executor_llm = None
_validate_tool = None
_solve_tool = None


def load_server_config(config_path: str = None):
    """Load configuration and initialize LLMs."""
    global _config, _modeller_llm, _checker_llm, _validate_tool, _solve_tool
    global _formulator_llm, _equation_checker_llm, _translator_llm, _code_checker_llm, _solver_executor_llm
    
    try:
        _config = load_config(config_path)
        
        # Load LLMs for simple 2-agent workflow
        _modeller_llm, _checker_llm = create_llms_from_config(config_path)
        
        # Load LLMs for complex 5-agent workflow
        (_formulator_llm, _equation_checker_llm, _translator_llm, 
         _code_checker_llm, _solver_executor_llm) = create_complex_workflow_llms(config_path)
        
        _validate_tool = create_validate_minizinc_tool()
        
        # Get solver backend from config (default to coinbc)
        workflow_config = _config.get("workflow", {})
        solver_backend = workflow_config.get("solver_backend", "coinbc")
        # Map 'mzn' to 'coinbc' for backwards compatibility
        if solver_backend == "mzn":
            solver_backend = "coinbc"
        
        _solve_tool = create_solve_minizinc_tool(
            solver_factory=lambda: MiniZincSolver(solver_backend=solver_backend)
        )
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        print(f"Config path: {config_path or 'default (repo_root/config.yaml)'}", file=sys.stderr)
        print(f"Working directory: {Path.cwd()}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


async def main_stdio(config_path: str = None):
    """Run MCP server with stdio transport."""
    from fastmcp import FastMCP
    
    load_server_config(config_path)
    
    # Create FastMCP server
    mcp_app = FastMCP("modeller-checker")
    
    # Register tool with wrapper for sync invoke
    class SyncToolWrapper:
        """Wraps sync tool to work in async context."""
        def __init__(self, tool):
            self.tool = tool
        
        async def invoke_async(self, input_dict):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.tool.invoke(input_dict))
    
    async_solve_tool = SyncToolWrapper(_solve_tool)
    
    @mcp_app.tool()
    async def modeller_checker_workflow(problem: str, max_iterations: int = None) -> str:
        """
        Dual-agent workflow for optimization problem modeling.
        
        Args:
            problem: Natural language problem description
            max_iterations: Max refinement iterations (default from config)
        
        Returns:
            Solution with MiniZinc model and optimal values
        """
        workflow_config = _config.get("workflow", {})
        verbose = workflow_config.get("verbose", False)
        # Use provided max_iterations or fall back to config value or default to 5
        iterations = max_iterations if max_iterations is not None else workflow_config.get("max_iterations", 5)
        
        result = await run_modeller_checker_workflow(
            problem=problem,
            modeller_llm=_modeller_llm,
            checker_llm=_checker_llm,
            validate_tool=_validate_tool,
            solve_tool=async_solve_tool,
            max_iterations=iterations,
            verbose=verbose,
        )
        
        response = f"""Success: {result['success']}
Checker Approval: {result['checker_approval']}
Iterations: {result['iterations']}

{result['final_response']}
"""
        
        if result['mzn_code']:
            response += f"\nMiniZinc Model:\n{result['mzn_code']}"
        
        return response
    
    @mcp_app.tool()
    async def complex_workflow(problem: str, max_iterations: int = None) -> str:
        """
        Complex 5-agent workflow for optimization problem modeling.
        
        Uses specialized agents for:
        - Formulation: Problem → Mathematical equations
        - Equation checking: Validates equations match problem
        - Translation: Equations → MiniZinc code
        - Code checking: Validates MiniZinc implementation
        - Solver execution: Runs solver and diagnoses errors
        
        Args:
            problem: Natural language problem description
            max_iterations: Max refinement iterations (default from config)
        
        Returns:
            Solution with formulation, MiniZinc model, and optimal values
        """
        workflow_config = _config.get("workflow", {})
        verbose = workflow_config.get("verbose", False)
        iterations = max_iterations if max_iterations is not None else workflow_config.get("max_iterations", 10)
        
        result = await run_complex_workflow(
            problem=problem,
            formulator_llm=_formulator_llm,
            equation_checker_llm=_equation_checker_llm,
            translator_llm=_translator_llm,
            code_checker_llm=_code_checker_llm,
            solver_executor_llm=_solver_executor_llm,
            validate_tool=_validate_tool,
            solve_tool=async_solve_tool,
            max_iterations=iterations,
            verbose=verbose,
        )
        
        response = f"""Success: {result['success']}
Iterations: {result['iterations']}
Workflow Trace: {' -> '.join(result['workflow_trace'])}

{result['final_response']}
"""
        
        if result.get('formulation'):
            from modeller_checker.complex_workflow import format_formulation_for_display
            response += f"\n{'='*60}\nMATHEMATICAL FORMULATION\n{'='*60}\n"
            response += format_formulation_for_display(result['formulation'])
        
        if result.get('mzn_code'):
            response += f"\n\n{'='*60}\nMINIZINC MODEL\n{'='*60}\n{result['mzn_code']}"
        
        return response
    
    await mcp_app.run_stdio_async(show_banner=False)


async def main_http(port: int = None, host: str = None, config_path: str = None):
    """Run MCP server with HTTP transport."""
    from fastmcp import FastMCP
    
    load_server_config(config_path)
    
    # Get HTTP settings from config if not provided via CLI
    mcp_server_config = _config.get("mcp_server", {})
    if port is None:
        port = mcp_server_config.get("http_port", 8767)
    if host is None:
        host = mcp_server_config.get("http_host", "127.0.0.1")
    
    # Create FastMCP server
    mcp_app = FastMCP("modeller-checker")
    
    # Register tool with wrapper for sync invoke
    class SyncToolWrapper:
        """Wraps sync tool to work in async context."""
        def __init__(self, tool):
            self.tool = tool
        
        async def invoke_async(self, input_dict):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.tool.invoke(input_dict))
    
    async_solve_tool = SyncToolWrapper(_solve_tool)
    
    @mcp_app.tool()
    async def modeller_checker_workflow(problem: str, max_iterations: int = None) -> str:
        """
        Dual-agent workflow for optimization problem modeling.
        
        Args:
            problem: Natural language problem description
            max_iterations: Max refinement iterations (default from config)
        
        Returns:
            Solution with MiniZinc model and optimal values
        """
        workflow_config = _config.get("workflow", {})
        verbose = workflow_config.get("verbose", False)
        # Use provided max_iterations or fall back to config value or default to 5
        iterations = max_iterations if max_iterations is not None else workflow_config.get("max_iterations", 5)
        
        result = await run_modeller_checker_workflow(
            problem=problem,
            modeller_llm=_modeller_llm,
            checker_llm=_checker_llm,
            validate_tool=_validate_tool,
            solve_tool=async_solve_tool,
            max_iterations=iterations,
            verbose=verbose,
        )
        
        response = f"""Success: {result['success']}
Checker Approval: {result['checker_approval']}
Iterations: {result['iterations']}

{result['final_response']}
"""
        
        if result['mzn_code']:
            response += f"\nMiniZinc Model:\n{result['mzn_code']}"
        
        return response
    
    @mcp_app.tool()
    async def complex_workflow(problem: str, max_iterations: int = None) -> str:
        """
        Complex 5-agent workflow for optimization problem modeling.
        
        Uses specialized agents for:
        - Formulation: Problem → Mathematical equations
        - Equation checking: Validates equations match problem
        - Translation: Equations → MiniZinc code
        - Code checking: Validates MiniZinc implementation
        - Solver execution: Runs solver and diagnoses errors
        
        Args:
            problem: Natural language problem description
            max_iterations: Max refinement iterations (default from config)
        
        Returns:
            Solution with formulation, MiniZinc model, and optimal values
        """
        workflow_config = _config.get("workflow", {})
        verbose = workflow_config.get("verbose", False)
        iterations = max_iterations if max_iterations is not None else workflow_config.get("max_iterations", 10)
        
        result = await run_complex_workflow(
            problem=problem,
            formulator_llm=_formulator_llm,
            equation_checker_llm=_equation_checker_llm,
            translator_llm=_translator_llm,
            code_checker_llm=_code_checker_llm,
            solver_executor_llm=_solver_executor_llm,
            validate_tool=_validate_tool,
            solve_tool=async_solve_tool,
            max_iterations=iterations,
            verbose=verbose,
        )
        
        response = f"""Success: {result['success']}
Iterations: {result['iterations']}
Workflow Trace: {' -> '.join(result['workflow_trace'])}

{result['final_response']}
"""
        
        if result.get('formulation'):
            from modeller_checker.complex_workflow import format_formulation_for_display
            response += f"\n{'='*60}\nMATHEMATICAL FORMULATION\n{'='*60}\n"
            response += format_formulation_for_display(result['formulation'])
        
        if result.get('mzn_code'):
            response += f"\n\n{'='*60}\nMINIZINC MODEL\n{'='*60}\n{result['mzn_code']}"
        
        return response
    
    await mcp_app.run_http_async(
        transport="streamable-http",
        host=host,
        port=port,
        path="/mcp",
        show_banner=False,
    )


def main():
    """CLI entry point."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    parser = argparse.ArgumentParser(
        description="MCP Server for Modeller-Checker Workflow"
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport (for Ollama integration)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP transport",
    )
    parser.add_argument(
        "--http-host",
        type=str,
        default=None,
        help="HTTP server host (default from config: 127.0.0.1)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=None,
        help="HTTP server port (default from config: 8767)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml (default: repo_root/config.yaml)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        # Enable debug logging for our modules only
        logging.getLogger("modeller_checker").setLevel(logging.DEBUG)
        logging.getLogger("mzn").setLevel(logging.DEBUG)
        logging.getLogger("mathprog").setLevel(logging.DEBUG)
        logging.getLogger("langchain_optimise").setLevel(logging.DEBUG)
        logging.getLogger("mcp.server").setLevel(logging.DEBUG)
        # Suppress verbose logging from dependencies
        logging.getLogger("fakeredis").setLevel(logging.WARNING)
        logging.getLogger("docket.worker").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        print(f"DEBUG: Working directory: {Path.cwd()}", file=sys.stderr)
        print(f"DEBUG: Python path: {sys.path[:3]}", file=sys.stderr)
        print(f"DEBUG: Config path: {args.config or 'default'}", file=sys.stderr)
    else:
        # Suppress debug from dependencies in normal mode
        logging.getLogger("fakeredis").setLevel(logging.WARNING)
        logging.getLogger("docket.worker").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    if args.stdio and args.http:
        print("Error: Cannot use both --stdio and --http", file=sys.stderr)
        sys.exit(1)
    
    if not args.stdio and not args.http:
        # Default to stdio
        args.stdio = True
    
    try:
        if args.stdio:
            asyncio.run(main_stdio(args.config))
        else:
            asyncio.run(main_http(args.http_port, args.http_host, args.config))
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
