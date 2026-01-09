# Integration Guide

Complete guide for integrating optimise-mcp components into your applications via MCP servers, LangChain tools, or direct Python APIs.

## Overview

Optimise-mcp provides three main components, each accessible via multiple integration methods:

1. **LP/MILP Solver** - Linear and mixed-integer programming
2. **MiniZinc Solver** - Constraint programming
3. **Modeller-Checker** - Dual-agent AI workflow

Each component can be used via:
- MCP Server (stdio or HTTP)
- LangChain Tool
- Direct Python API
- CLI Script

## Configuration

All components use a single `config.yaml` file in the repo root:

```yaml
# LP/MILP Solver Configuration
mathprog:
  solver:
    backend: "highs"
    time_limit: 60
  mcp_server:
    http_port: 8765
    log_level: "INFO"

# MiniZinc Solver Configuration
mzn:
  solver:
    backend: "coinbc"  # or gecode, chuffed, or-tools
    use_python_bindings: true
    time_limit: 60
  mcp_server:
    http_port: 8766
    log_level: "INFO"

# Modeller-Checker Workflow Configuration
modeller_checker:
  modeller:
    provider: "ollama"  # or openai, anthropic, azure
    model: "qwen3"
    base_url: "http://127.0.0.1:11434"
    temperature: 0.5
  checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  workflow:
    max_iterations: 5
  mcp_server:
    http_port: 8767
```

## 1. LP/MILP Solver Integration

### MCP Server (Stdio)

For Ollama, Claude Desktop, and other stdio MCP clients:

```bash
python -m src.mathprog.mcp --stdio
```

MCP client configuration:
```json
{
  "mcpServers": {
    "lp-solver": {
      "command": "python",
      "args": ["-m", "src.mathprog.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    }
  }
}
```

### MCP Server (HTTP)

```bash
python -m src.mathprog.mcp --http --http-port 8765
```

### LangChain Tool

```python
from langchain_optimise.lp_tools import create_validate_lp_tool, create_solve_lp_tool

# Create tools
validate_lp = create_validate_lp_tool()
solve_lp = create_solve_lp_tool()

# Use in your agent
lp_code = """
Maximize
  40 x + 30 y
Subject To
  land: x + y <= 110
  labor: 3 x + 2 y <= 240
Bounds
  x >= 0
  y >= 0
End
"""

# Validate
validation_result = validate_lp.invoke({"lp_code": lp_code})

# Solve
solution = solve_lp.invoke({"lp_code": lp_code, "time_limit": 60})
print(solution)
```

### Direct Python API

```python
from lp.validator import validate_lp
from lp.solver import solve_lp

# Validate
is_valid, issues = validate_lp(lp_code)

# Solve
result = solve_lp(lp_code, time_limit=60)
```

## 2. MiniZinc Solver Integration

### MCP Server (Stdio)

```bash
python -m src.mzn.mcp --stdio
```

MCP client configuration:
```json
{
  "mcpServers": {
    "minizinc-solver": {
      "command": "python",
      "args": ["-m", "src.mzn.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    }
  }
}
```

### MCP Server (HTTP)

```bash
python -m src.mzn.mcp --http --http-port 8766
```

### LangChain Tool

```python
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool
)

# Create tools
validate_mzn = create_validate_minizinc_tool()
solve_mzn = create_solve_minizinc_tool()

# MiniZinc code
mzn_code = """
var int: x;
var int: y;

constraint x + y <= 110;
constraint 3*x + 2*y <= 240;
constraint x >= 0;
constraint y >= 0;

solve maximize 40*x + 30*y;
"""

# Validate
validation = validate_mzn.invoke({"mzn_code": mzn_code})

# Solve
solution = solve_mzn.invoke({"mzn_code": mzn_code, "time_limit": 60})
print(solution)
```

### Direct Python API

```python
from minizinc.validator import validate_minizinc
from minizinc.solver import solve_minizinc

# Validate
validation_result = validate_minizinc(mzn_code)

# Solve
solution = solve_minizinc(mzn_code, solver_backend="coinbc", time_limit=60)
```

## 3. Modeller-Checker Integration

The modeller-checker uses two separate LLM instances (configurable in `config.yaml`).

### MCP Server (Stdio)

```bash
python -m src.modeller_checker.mcp --stdio
```

MCP client configuration:
```json
{
  "mcpServers": {
    "modeller-checker": {
      "command": "python",
      "args": ["-m", "src.modeller_checker.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    }
  }
}
```

### MCP Server (HTTP)

```bash
python -m src.modeller_checker.mcp --http --http-port 8767
```

### LangChain Tool

```python
from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool

# Create tool (loads config.yaml automatically)
tool = create_modeller_checker_tool(verbose=True)

# Use in your agent
problem = """
We have 110 acres of land. We can plant wheat or corn.
Wheat yields $40 profit per acre and requires 3 labour hours per acre.
Corn yields $30 profit per acre and requires 2 labour hours per acre.
We have 240 labour hours available.
How should we allocate the land to maximize profit?
"""

result = tool.invoke({
    "problem": problem,
    "max_iterations": 5
})

print(result)
```

### CLI Script

```bash
# Default problem
python scripts/langchain_modeller_checker.py -v

# Custom problem
python scripts/langchain_modeller_checker.py -v \
  -p "Maximize x+y subject to x+y <= 100, x >= 0, y >= 0" \
  --iterations 3

# Custom config
python scripts/langchain_modeller_checker.py -v \
  --config my_config.yaml
```

### Direct Python API

```python
from modeller_checker.config import create_llms_from_config
from modeller_checker.workflow import run_modeller_checker_workflow
from langchain_optimise.minizinc_tools import (
    create_validate_minizinc_tool,
    create_solve_minizinc_tool
)

# Load LLMs from config
modeller_llm, checker_llm = create_llms_from_config()

# Create tools
validate_tool = create_validate_minizinc_tool()
solve_tool = create_solve_minizinc_tool()

# Run workflow
import asyncio

result = asyncio.run(run_modeller_checker_workflow(
    problem="Your problem description",
    modeller_llm=modeller_llm,
    checker_llm=checker_llm,
    validate_tool=validate_tool,
    solve_tool=solve_tool,
    max_iterations=5,
    verbose=True
))

print(result["final_response"])
print(result["mzn_code"])
```

## Cloud Provider Configuration

### Using Different LLMs for Modeller and Checker

You can mix local and cloud models:

```yaml
modeller_checker:
  modeller:
    provider: "ollama"
    model: "qwen3"
    base_url: "http://127.0.0.1:11434"
    temperature: 0.5
  
  checker:
    provider: "openai"
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"  # Reads from env var
    temperature: 0.3
    max_tokens: 1500
```

### OpenAI

```yaml
modeller:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.5
  max_tokens: 2000
```

Requires: `pip install langchain-openai`

### Anthropic

```yaml
checker:
  provider: "anthropic"
  model: "claude-3-sonnet-20240229"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.3
  max_tokens: 1500
```

Requires: `pip install langchain-anthropic`

### Azure OpenAI

```yaml
modeller:
  provider: "azure"
  model: "gpt-4"
  api_key: "${AZURE_OPENAI_API_KEY}"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  api_version: "2024-02-15-preview"
  temperature: 0.5
```

Requires: `pip install langchain-openai`

## Solver Backend Configuration

### MiniZinc Solvers

Available solvers (if installed):
- `coinbc` - Coin-OR Branch and Cut (default, MILP)
- `gecode` - Gecode (constraint programming)
- `chuffed` - Lazy clause generation
- `or-tools` - Google OR-Tools

To see available solvers:
```bash
minizinc --solvers
```

Configure in `config.yaml`:
```yaml
minizinc:
  solver:
    backend: "gecode"  # Change here
    time_limit: 120
```

### LP Solver

Currently only HiGHS is supported:
```yaml
lp:
  solver:
    backend: "highs"
    time_limit: 60
```

## Port Configuration

Each MCP server can run on HTTP with configurable ports:

```yaml
lp:
  mcp_server:
    http_port: 8765

minizinc:
  mcp_server:
    http_port: 8766

modeller_checker:
  mcp_server:
    http_port: 8767
```

Start servers:
```bash
python -m src.mathprog.mcp --http --http-port 8765 &
python -m src.mzn.mcp --http --http-port 8766 &
python -m src.modeller_checker.mcp --http --http-port 8767 &
```

**HTTP Endpoints:**
- LP/MILP: `http://127.0.0.1:8765/mcp`
- MiniZinc: `http://127.0.0.1:8766/mcp`
- Modeller-Checker: `http://127.0.0.1:8767/mcp`

The MCP protocol is exposed at the `/mcp` path (not the root `/`).

## Troubleshooting

**Config file not found:**
- Ensure `config.yaml` exists in repo root
- Or specify: `--config /path/to/config.yaml`

**API key errors:**
- Use environment variables: `api_key: "${OPENAI_API_KEY}"`
- Set env var before running: `export OPENAI_API_KEY=sk-...`

**Import errors for cloud providers:**
- Install required packages:
  - `pip install langchain-openai` for OpenAI/Azure
  - `pip install langchain-anthropic` for Anthropic

**Solver backend not found:**
- Install MiniZinc and desired solvers
- Check available: `minizinc --solvers`
- Update `config.yaml` to use available solver

**Port already in use:**
- Change port in `config.yaml`
- Or kill existing process: `lsof -ti:8765 | xargs kill`

## Testing Your Integration

```bash
# Test LP solver
python examples/lp_example.py

# Test MiniZinc solver
python examples/minizinc_example.py

# Test modeller-checker
python examples/modeller_checker_example.py

# Test all MCP servers (stdio)
python scripts/mcp_lp_roundtrip.py
python scripts/langchain_minizinc_roundtrip.py
python scripts/langchain_modeller_checker.py -v
```

## See Also

- [README.md](../README.md) - Main documentation
- [MODELLER_CHECKER.md](MODELLER_CHECKER.md) - Workflow details
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guide
- [TESTING.md](TESTING.md) - Testing guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference

1. **MCP Server** (stdio or HTTP) - for MCP clients like Claude Desktop, Ollama
2. **LangChain Tool** - for direct Python integration in LangChain agents
3. **CLI Script** - for command-line usage

All three methods use the same configuration file for consistent LLM settings.

## Configuration

Create `config.yaml` in the repo root:

```yaml
# Modeller LLM Configuration
modeller:
  provider: "ollama"  # Options: ollama, openai, anthropic, azure
  model: "qwen3"
  base_url: "http://127.0.0.1:11434"
  temperature: 0.5

# Checker LLM Configuration  
checker:
  provider: "ollama"
  model: "qwen3"  # Can use different model than modeller
  base_url: "http://127.0.0.1:11434"
  temperature: 0.3

# Workflow Settings
workflow:
  max_iterations: 5
  verbose: false
```

### Using Different LLMs

You can mix local and cloud models:

```yaml
# Example: Local modeller, cloud checker
modeller:
  provider: "ollama"
  model: "qwen3"
  base_url: "http://127.0.0.1:11434"
  temperature: 0.5

checker:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"  # Reads from environment variable
  temperature: 0.3
```

Supported providers:
- **ollama** - Local models via Ollama
- **openai** - OpenAI API (requires `pip install langchain-openai`)
- **anthropic** - Anthropic/Claude API (requires `pip install langchain-anthropic`)
- **azure** - Azure OpenAI (requires `pip install langchain-openai`)

## Usage Methods

### 1. MCP Server (Stdio)

For integration with Ollama and Claude Desktop:

```bash
# Start MCP server with stdio transport
python -m src.modeller_checker.mcp --stdio

# Or with custom config
python -m src.modeller_checker.mcp --stdio --config my_config.yaml
```

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "modeller-checker": {
      "command": "python",
      "args": ["-m", "src.modeller_checker.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    }
  }
}
```

### 2. MCP Server (HTTP)

For HTTP-based MCP clients:

```bash
# Start HTTP server on port 8767
python -m src.modeller_checker.mcp --http --http-port 8767

# Or with custom config
python -m src.modeller_checker.mcp --http --http-port 8767 --config my_config.yaml
```

### 3. LangChain Tool

For direct Python integration:

```python
from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool

# Create tool (loads config.yaml automatically)
tool = create_modeller_checker_tool(verbose=True)

# Use in LangChain agent
result = tool.invoke({
    "problem": (
        "We have 110 acres of land. We can plant wheat or corn. "
        "Wheat yields $40 profit per acre and requires 3 labour hours per acre. "
        "Corn yields $30 profit per acre and requires 2 labour hours per acre. "
        "We have 240 labour hours available. "
        "How should we allocate the land to maximize profit?"
    ),
    "max_iterations": 5
})

print(result)
```

### 4. CLI Script

For command-line usage:

```bash
# Use default problem and config.yaml
python scripts/langchain_modeller_checker.py -v

# Custom problem
python scripts/langchain_modeller_checker.py -v \
  -p "Maximize x+y subject to x+y <= 100, x >= 0, y >= 0"

# Custom config file
python scripts/langchain_modeller_checker.py -v \
  --config my_config.yaml \
  --iterations 3

# Simple test
python scripts/langchain_modeller_checker.py \
  -p "Maximize x subject to x <= 10"
```

## MCP Tool Schema

When using via MCP, the tool is exposed as:

```json
{
  "name": "modeller_checker_workflow",
  "description": "Dual-agent workflow for optimization problem modeling...",
  "inputSchema": {
    "type": "object",
    "properties": {
      "problem": {
        "type": "string",
        "description": "Natural language problem description"
      },
      "max_iterations": {
        "type": "integer",
        "description": "Max modeller-checker refinement iterations",
        "default": 5
      }
    },
    "required": ["problem"]
  }
}
```

## Example Problems

### Farm Allocation (Default)
```
We have 110 acres of land. We can plant wheat or corn. 
Wheat yields $40 profit per acre and requires 3 labour hours per acre. 
Corn yields $30 profit per acre and requires 2 labour hours per acre. 
We have 240 labour hours available. 
How should we allocate the land to maximize profit?
```

### Bakery Production
```
A bakery makes bread and cakes. Bread takes 30 min to make and yields $5 profit. 
Cakes take 1 hour and yield $8 profit. The bakery has 8 hours per day available. 
They can make at most 20 breads per day. What should they bake to maximize profit?
```

### Simple Test
```
Maximize x subject to x <= 10
```

## Workflow Architecture

```
Problem Description
        ↓
    [MODELLER LLM]  → Creates MiniZinc Model
        ↓
   [Syntax Validation]
        ↓
    [CHECKER LLM]  → Verifies Model vs Problem
        ↓
    ┌─────────────────────┐
    │ Approved? Yes       │  No
    └─────────────────────┘
         ↓                 ↓
      [SOLVER]        [Feedback to Modeller]
         ↓                 ↓
      Solution       (Loop back to Modeller)
```

The modeller and checker use separate LLM instances (configurable), allowing you to:
- Use stronger models for checking
- Use faster models for generation
- Mix local and cloud providers
- Control temperature independently

## Files

- `config.yaml` - Configuration file (create in repo root)
- `src/modeller_checker/workflow.py` - Core workflow logic
- `src/modeller_checker/config.py` - Config loader and LLM factory
- `src/modeller_checker/mcp.py` - MCP server (stdio + HTTP)
- `src/langchain_optimise/modeller_checker_tool.py` - LangChain tool factory
- `scripts/langchain_modeller_checker.py` - CLI wrapper

## Testing

```bash
# Test CLI
python scripts/langchain_modeller_checker.py -v

# Test with simple problem
python scripts/langchain_modeller_checker.py -p "Maximize x subject to x <= 10"

# Test MCP server (in another terminal, send MCP requests)
python -m src.modeller_checker.mcp --stdio

# Test LangChain tool
python -c "
from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool
tool = create_modeller_checker_tool()
result = tool.invoke({'problem': 'Maximize x subject to x <= 10'})
print(result)
"
```

## Troubleshooting

**Config file not found:**
- Ensure `config.yaml` exists in repo root
- Or specify path: `--config /path/to/config.yaml`

**API key errors:**
- Use environment variables: `api_key: "${OPENAI_API_KEY}"`
- Ensure env var is set before running

**Import errors:**
- Install cloud provider packages:
  - `pip install langchain-openai` for OpenAI/Azure
  - `pip install langchain-anthropic` for Anthropic

**Model generates invalid syntax:**
- Try a different model (qwen3 works well)
- Lower temperature for more deterministic output
- Increase max_iterations for more refinement rounds
