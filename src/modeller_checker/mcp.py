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

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from modeller_checker.config import load_config, create_llms_from_config
from modeller_checker.workflow import run_modeller_checker_workflow
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool,
)


# Global server instance
app = Server("modeller-checker")

# Global config and LLMs (loaded on server start)
_config = None
_modeller_llm = None
_checker_llm = None
_validate_tool = None
_solve_tool = None


def load_server_config(config_path: str = None):
    """Load configuration and initialize LLMs."""
    global _config, _modeller_llm, _checker_llm, _validate_tool, _solve_tool
    
    try:
        _config = load_config(config_path)
        _modeller_llm, _checker_llm = create_llms_from_config(config_path)
        _validate_tool = create_validate_minizinc_tool()
        _solve_tool = create_solve_minizinc_tool()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        print(f"Config path: {config_path or 'default (repo_root/config.yaml)'}", file=sys.stderr)
        print(f"Working directory: {Path.cwd()}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="modeller_checker_workflow",
            description=(
                "Dual-agent workflow for optimization problem modeling. "
                "A Modeller agent creates MiniZinc models from problem descriptions, "
                "a Checker agent validates correctness, and feedback loops refine the model. "
                "Once approved, the model is solved and the optimal solution is returned. "
                "Use this for complex optimization problems requiring validation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": (
                            "Natural language problem description. "
                            "Should include: decision variables, constraints, objective function, "
                            "and any resource limits or requirements."
                        ),
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Max modeller-checker refinement iterations (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["problem"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name != "modeller_checker_workflow":
        raise ValueError(f"Unknown tool: {name}")
    
    problem = arguments.get("problem")
    if not problem:
        raise ValueError("'problem' argument is required")
    
    max_iterations = arguments.get("max_iterations", 5)
    verbose = _config.get("workflow", {}).get("verbose", False)
    
    # Run workflow
    result = await run_modeller_checker_workflow(
        problem=problem,
        modeller_llm=_modeller_llm,
        checker_llm=_checker_llm,
        validate_tool=_validate_tool,
        solve_tool=_solve_tool,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    
    # Format response
    response_text = f"""Modeller-Checker Workflow Results:

Success: {result['success']}
Checker Approval: {result['checker_approval']}
Iterations: {result['iterations']}

Final Response:
{result['final_response']}
"""
    
    if result['mzn_code']:
        response_text += f"\nFinal MiniZinc Model:\n{result['mzn_code']}"
    
    return [TextContent(type="text", text=response_text)]


async def main_stdio(config_path: str = None):
    """Run MCP server with stdio transport."""
    load_server_config(config_path)
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main_http(port: int = 8767, config_path: str = None):
    """Run MCP server with HTTP transport."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("FastMCP not installed. Install with: pip install mcp[server]", file=sys.stderr)
        sys.exit(1)
    
    load_server_config(config_path)
    
    # Create FastMCP wrapper
    mcp_app = FastMCP("modeller-checker")
    
    # Configure port
    mcp_app.settings.port = port
    
    # Register tool
    @mcp_app.tool()
    async def modeller_checker_workflow(problem: str, max_iterations: int = 5) -> str:
        """
        Dual-agent workflow for optimization problem modeling.
        
        Args:
            problem: Natural language problem description
            max_iterations: Max refinement iterations
        
        Returns:
            Solution with MiniZinc model and optimal values
        """
        verbose = _config.get("workflow", {}).get("verbose", False)
        
        result = await run_modeller_checker_workflow(
            problem=problem,
            modeller_llm=_modeller_llm,
            checker_llm=_checker_llm,
            validate_tool=_validate_tool,
            solve_tool=_solve_tool,
            max_iterations=max_iterations,
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
    
    print(f"Starting MCP server on http://{mcp_app.settings.host}:{port}", file=sys.stderr)
    mcp_app.run(transport="streamable-http")


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
        "--http-port",
        type=int,
        default=8767,
        help="HTTP server port (default: 8767)",
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
        logging.getLogger().setLevel(logging.DEBUG)
        print(f"DEBUG: Working directory: {Path.cwd()}", file=sys.stderr)
        print(f"DEBUG: Python path: {sys.path[:3]}", file=sys.stderr)
        print(f"DEBUG: Config path: {args.config or 'default'}", file=sys.stderr)
    
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
            main_http(args.http_port, args.config)
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
