# optimise-mcp

AI-powered optimization problem solver using a 5-agent workflow. Automatically converts natural language problem descriptions into MiniZinc code and solves them.

## Features

### 🎯 5-Agent Optimization Workflow

Transform optimization problems from natural language to solutions:

**Problem** → **Formulator** → **Equation Checker** → **Translator** → **Code Checker** → **Solver Executor** → **Solution**

#### The 5 Specialized Agents:

1. **Formulator**: Converts natural language into mathematical equations
2. **Equation Checker**: Validates mathematical correctness and completeness
3. **Translator**: Converts equations into MiniZinc constraint programming code
4. **Code Checker**: Validates MiniZinc syntax and semantics
5. **Solver Executor**: Runs the solver and diagnoses errors

**Key Capabilities:**
- **Intelligent error routing** - Errors automatically routed to the right agent
- **Multi-stage validation** - Two validation checkpoints ensure correctness
- **Explicit formulation** - Mathematical representation before code generation
- **Multi-provider support** - Mix local (Ollama) and cloud (OpenAI, Anthropic, Azure) LLMs
- **Three integration paths** - MCP server, LangChain tool, or CLI

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for detailed documentation.

## Quick Start

### Installation
```bash
pip install -e .
```

### Prerequisites
- Python 3.11+
- MiniZinc (install from https://www.minizinc.org/software.html)
- Ollama (for local LLMs) or API keys for cloud providers

### Configuration

Create `config.yaml`:

```yaml
modeller_checker:
  # 5-agent workflow - each agent configurable
  formulator:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.5
  equation_checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  translator:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.4
  code_checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.2
  solver_executor:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  
  workflow:
    max_iterations: 10
  mcp_server:
    http_port: 8767

# MiniZinc solver backend (used by workflow)
mzn:
  solver:
    backend: "coinbc"  # or gecode, chuffed, or-tools
    time_limit: 60
```

### Usage Examples

#### CLI Test
```bash
python scripts/workflow_test.py -v -p "Allocate 100 hours between product A (profit \$50/unit, 2 hours each) and product B (profit \$30/unit, 1 hour each) to maximize profit"
```

#### LangChain Tool
```python
from langchain_optimise.workflow_tool import create_optimization_workflow_tool

# Create the tool
workflow_tool = create_optimization_workflow_tool()

# Use with an agent
problem = """
A factory produces two products A and B. 
Product A gives $50 profit and takes 2 hours.
Product B gives $30 profit and takes 1 hour.
We have 100 hours available. How many of each should we make?
"""

result = workflow_tool.invoke({"problem": problem})
print(result)
```

#### MCP Server
```bash
# Start MCP server
python -m src.modeller_checker.mcp --stdio    # stdio transport
python -m src.modeller_checker.mcp --http     # HTTP on port 8767
```

## Project Structure

```
optimise-mcp/
├── src/
│   ├── modeller_checker/      # 5-agent workflow
│   │   ├── workflow.py         # Main workflow orchestration
│   │   ├── config.py           # Configuration loading
│   │   └── mcp.py              # MCP server
│   ├── langchain_optimise/     # LangChain tools
│   │   ├── workflow_tool.py    # Workflow tool wrapper
│   │   └── minizinc_tools.py   # MiniZinc tools (used by workflow)
│   └── mzn/                    # MiniZinc solver utilities
│       ├── solver.py
│       └── validator.py
├── scripts/
│   └── workflow_test.py        # CLI test script
├── tests/
│   └── test_minizinc_solver.py
├── docs/
│   ├── WORKFLOW.md             # Main workflow documentation
│   ├── QUICK_REFERENCE.md      # Quick reference
│   ├── MIGRATION_GUIDE.md      # Migration from older versions
│   ├── CONFIGURATION.md        # Configuration details
│   ├── INTEGRATION_GUIDE.md    # Integration options
│   └── TESTING.md              # Testing guide
└── config.yaml                 # Configuration file
```

## Configuration Guide

The system is configured via `config.yaml`:

### Workflow Configuration
```yaml
modeller_checker:
  # Configure each of the 5 agents independently
  formulator:
    provider: "ollama"        # ollama, openai, anthropic, azure
    model: "qwen3"
    temperature: 0.5
  equation_checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  translator:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.4
  code_checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.2
  solver_executor:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  workflow:
    max_iterations: 10
  mcp_server:
    http_port: 8767
```

### MiniZinc Configuration
```yaml
mzn:
  solver:
    backend: "coinbc"         # coinbc, gecode, chuffed, or-tools
    use_python_bindings: true # false = CLI fallback
    time_limit: 60
```

**Cloud Provider Support:**
- OpenAI: Requires `pip install langchain-openai`
- Anthropic: Requires `pip install langchain-anthropic`
- Azure: Requires `pip install langchain-openai`

See [config.yaml](config.yaml) for detailed examples.

## MCP Integration

### Stdio Transport (Ollama, Claude Desktop)

Add to MCP client configuration:

```json
{
  "mcpServers": {
    "optimization-workflow": {
      "command": "python",
      "args": ["-m", "src.modeller_checker.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    }
  }
}
```

### HTTP Transport

```bash
# Start server
python -m src.modeller_checker.mcp --http --http-port 8767
```

**Note:** HTTP server exposes the MCP endpoint at `/mcp`. Connect to: `http://127.0.0.1:8767/mcp`

## Testing

```bash
# Run tests
pytest tests/test_minizinc_solver.py -v

# Test workflow
python scripts/workflow_test.py -v -p "Your optimization problem"
```

## Documentation

- [WORKFLOW.md](docs/WORKFLOW.md) - Complete workflow guide
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Quick reference
- [MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) - Migrating from older versions
- [CONFIGURATION.md](docs/CONFIGURATION.md) - Configuration details
- [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) - Integration patterns
- [TESTING.md](docs/TESTING.md) - Testing guide

## Requirements

Core dependencies:
- `langchain>=0.3.0`
- `langchain-ollama>=0.1.0`
- `fastmcp>=2.14.0`
- `pydantic>=2.8`
- `pyyaml>=6.0`
- MiniZinc (external install required)

Optional dependencies:
- `langchain-openai` - For OpenAI/Azure
- `langchain-anthropic` - For Anthropic/Claude

## License

MIT

## Contributing

Contributions welcome! Please see [DEVELOPMENT.md](docs/DEVELOPMENT.md) for guidelines.
