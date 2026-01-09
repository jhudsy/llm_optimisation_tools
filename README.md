# optimise-mcp

Production-ready Model Context Protocol (MCP) server suite for optimization problems. Provides LP/MILP solving (HiGHS), constraint programming (MiniZinc), and a dual-agent AI workflow for problem modeling.

## Features

### 🎯 Three Core Functionalities

#### 1. **LP/MILP Solver** (`src/mathprog/`)
- **HiGHS solver backend** for linear and mixed-integer programs
- **Direct file support** - accepts `.lp` files
- **Validator-first design** catches issues early
- **Quadratic objectives** supported
- **MCP integration** via stdio or HTTP

#### 2. **MiniZinc Solver** (`src/mzn/`)
- **Constraint programming** for complex optimization
- **Direct file support** - accepts `.mzn` files  
- **Multiple solvers** - coinbc, gecode, chuffed, or-tools
- **Python + CLI** - bindings with fallback
- **MCP integration** via stdio or HTTP

#### 3. **Modeller-Checker Workflow** (`src/modeller_checker/`)
- **Dual-agent AI system** - separate modeller and checker LLMs
- **Automatic model generation** from problem descriptions
- **Validation & refinement** loop ensures correctness
- **Multi-provider support** - mix local (Ollama) and cloud (OpenAI, Anthropic, Azure) LLMs
- **Three integration paths** - MCP server, LangChain tool, CLI

## Quick Start

### Installation
```bash
pip install -e .
```

### Configuration

Create `config.yaml` (or edit the provided template):

```yaml
# LP/MILP Solver (Mathematical Programming)
mathprog:
  solver:
    backend: "highs"
    time_limit: 60
  mcp_server:
    http_port: 8765

# MiniZinc Constraint Programming Solver
mzn:
  solver:
    backend: "coinbc"
    time_limit: 60
  mcp_server:
    http_port: 8766

# Modeller-Checker (Dual-Agent AI)
modeller_checker:
  modeller:
    provider: "ollama"
    model: "qwen3"
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

### Usage Examples

#### LP/MILP Solver
```bash
# Start MCP server
python -m src.mathprog.mcp --stdio          # stdio transport
python -m src.mathprog.mcp --http           # HTTP on port 8765

# Test with LangChain tools
python -c "
from langchain_optimise.lp_tools import create_validate_lp_tool, create_solve_lp_tool
validate = create_validate_lp_tool()
solve = create_solve_lp_tool()

lp_code = '''
Maximize
 obj: x + 2 y
Subject To
 c1: x + y <= 10
Bounds
 0 <= x <= 10
 0 <= y <= 10
End
'''

result = solve.invoke({'lp_code': lp_code})
print(result)
"
```

#### MiniZinc Solver
```bash
# Start MCP server
python -m src.mzn.mcp --stdio    # stdio transport
python -m src.mzn.mcp --http     # HTTP on port 8766

# Run example
python examples/minizinc_example.py

# Test with .mzn file
python -c "
from langchain_optimise.minizinc_tools import create_solve_minizinc_tool
solve = create_solve_minizinc_tool()

with open('examples/test.mzn') as f:
    mzn_code = f.read()

result = solve.invoke({'mzn_code': mzn_code})
print(result)
"
```

#### Modeller-Checker Workflow
```bash
# Start MCP server
python -m src.modeller_checker.mcp --stdio    # stdio
python -m src.modeller_checker.mcp --http     # HTTP on port 8767

# CLI usage
python scripts/langchain_modeller_checker.py -v \
  -p "We have 110 acres. Plant wheat or corn to maximize profit..."

# LangChain tool
python examples/modeller_checker_example.py

# Python API
from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool
tool = create_modeller_checker_tool(verbose=True)
result = tool.invoke({
    "problem": "Maximize x+y subject to x+y<=100, x>=0, y>=0",
    "max_iterations": 5
})
```

## Project Structure

```
optimise_mcp/
├── config.yaml                 # Unified configuration
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── mathprog/              # LP/MILP solver (CPLEX .lp format)
│   │   ├── mcp.py            # MCP server
│   │   ├── solver.py         # HiGHS solver
│   │   └── validator.py      # LP validation
│   ├── mzn/                   # MiniZinc solver
│   │   ├── mcp.py            # MCP server
│   │   ├── solver.py         # MiniZinc solver
│   │   └── validator.py      # MiniZinc validation
│   ├── modeller_checker/      # Dual-agent workflow
│   │   ├── workflow.py       # Core logic
│   │   ├── config.py         # Config loader
│   │   └── mcp.py            # MCP server
│   └── langchain_optimise/    # LangChain tools
│       ├── lp_tools.py
│       ├── minizinc_tools.py
│       └── modeller_checker_tool.py
│
├── scripts/                    # CLI scripts
│   ├── mcp_lp_roundtrip.py
│   ├── langchain_minizinc_roundtrip.py
│   └── langchain_modeller_checker.py
│
├── examples/                   # Usage examples
│   ├── lp_example.py
│   ├── minizinc_example.py
│   └── modeller_checker_example.py
│
├── tests/                      # Test suite
│   ├── test_lp_tools.py
│   ├── test_minizinc_tools.py
│   └── ...
│
└── docs/                       # Documentation
    ├── DEVELOPMENT.md          # Development guide
    ├── TESTING.md              # Testing guide
    ├── MODELLER_CHECKER.md     # Workflow details
    └── INTEGRATION_GUIDE.md    # Integration options
```

## Configuration Guide

All three components are configured via `config.yaml`:

### LP/MILP Solver Configuration
```yaml
mathprog:
  solver:
    backend: "highs"           # Solver backend
    time_limit: 60            # Seconds
  mcp_server:
    stdio_enabled: true
    http_enabled: true
    http_port: 8765           # HTTP server port
    log_level: "INFO"
```

### MiniZinc Configuration
```yaml
mzn:
  solver:
    backend: "coinbc"         # coinbc, gecode, chuffed, or-tools
    use_python_bindings: true # false = CLI fallback
    time_limit: 60
  mcp_server:
    http_port: 8766
```

### Modeller-Checker Configuration
```yaml
modeller_checker:
  modeller:
    provider: "ollama"        # ollama, openai, anthropic, azure
    model: "qwen3"
    base_url: "http://127.0.0.1:11434"
    temperature: 0.5
  checker:
    provider: "ollama"        # Can differ from modeller
    model: "qwen3"
    temperature: 0.3
  workflow:
    max_iterations: 5
    solver_backend: "mzn"  # Use MiniZinc solver
  mcp_server:
    http_port: 8767
```

**Cloud Provider Support:**
- OpenAI: Requires `pip install langchain-openai`
- Anthropic: Requires `pip install langchain-anthropic`
- Azure: Requires `pip install langchain-openai`

See `config.yaml` for detailed examples.

## MCP Integration

### Stdio Transport (Ollama, Claude Desktop)

Add to MCP client configuration:

```json
{
  "mcpServers": {
    "lp-solver": {
      "command": "python",
      "args": ["-m", "src.mathprog.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    },
    "minizinc-solver": {
      "command": "python",
      "args": ["-m", "src.mzn.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    },
    "modeller-checker": {
      "command": "python",
      "args": ["-m", "src.modeller_checker.mcp", "--stdio"],
      "cwd": "/path/to/optimise_mcp"
    }
  }
}
```

### HTTP Transport

```bash
# Start all servers
python -m src.mathprog.mcp --http --http-port 8765 &
python -m src.mzn.mcp --http --http-port 8766 &
python -m src.modeller_checker.mcp --http --http-port 8767 &
```

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Test specific components
pytest tests/test_lp_tools.py -v
pytest tests/test_minizinc_tools.py -v

# Test roundtrip workflows
python scripts/mcp_lp_roundtrip.py
python scripts/langchain_minizinc_roundtrip.py
python scripts/langchain_modeller_checker.py -v
```

## Documentation

- **[docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)** - Complete integration guide
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** - Configuration reference
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guide  
- **[docs/TESTING.md](docs/TESTING.md)** - Testing guide
- **[docs/MODELLER_CHECKER.md](docs/MODELLER_CHECKER.md)** - Workflow details
- **[docs/example-prompts.md](docs/example-prompts.md)** - System prompts

## Key Differences Between Components

| Feature | LP Solver | MiniZinc Solver | Modeller-Checker |
|---------|-----------|-----------------|------------------|
| **Input** | `.lp` file or LP code | `.mzn` file or MiniZinc code | Natural language |
| **Solver** | HiGHS | Configurable (coinbc, gecode, etc.) | Uses MiniZinc internally |
| **Best For** | Linear/MILP problems | Constraint programming | Complex problems needing validation |
| **AI** | No | No | Yes (dual-agent) |
| **Validation** | Syntax + structure | Syntax + semantics | Problem → Model verification |

## License

MIT

## Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development setup and guidelines.
