# Configuration Guide

Complete reference for all configurable parameters in `config.yaml`.

## Overview

All components (LP solver, MiniZinc solver, Modeller-Checker) are configured via a single `config.yaml` file in the repository root. CLI arguments can override config file values.

## Configuration File Structure

```yaml
mathprog:                # LP/MILP solver settings
  solver: {...}
  mcp_server: {...}
  validation: {...}
  performance: {...}

mzn:                     # MiniZinc solver settings
  solver: {...}
  mcp_server: {...}
  validation: {...}
  performance: {...}

modeller_checker:        # Dual-agent workflow settings
  modeller: {...}
  checker: {...}
  workflow: {...}
  mcp_server: {...}
  llm_defaults: {...}

global:                  # Global settings
  log_level: "INFO"
  log_file: null
  debug_mode: false
```

## LP/MILP Solver Configuration

### `mathprog.solver`
- **`backend`** (string, default: `"highs"`)
  - LP solver backend
  - Currently only `"highs"` is supported
  
- **`time_limit`** (integer, default: `60`)
  - Maximum solver time in seconds
  - Can be overridden per-solve via API

### `mathprog.mcp_server`
- **`stdio_enabled`** (boolean, default: `true`)
  - Enable stdio MCP transport
  
- **`http_enabled`** (boolean, default: `true`)
  - Enable HTTP MCP transport
  
- **`http_host`** (string, default: `"127.0.0.1"`)
  - HTTP server bind address
  - CLI override: `--http-host`
  
- **`http_port`** (integer, default: `8765`)
  - HTTP server port
  - CLI override: `--http-port`
  
- **`log_level`** (string, default: `"INFO"`)
  - Logging level: DEBUG, INFO, WARNING, ERROR
  - CLI override: `--log-level`

### `mathprog.validation`
- **`max_variables`** (integer, default: `10000`)
  - Maximum number of variables allowed
  - Not yet enforced by validator
  
- **`max_constraints`** (integer, default: `10000`)
  - Maximum number of constraints allowed
  - Not yet enforced by validator

### `mathprog.performance`
- **`max_parallel_solvers`** (integer, default: `5`)
  - Maximum concurrent solver instances for LangChain tools
  - Controls parallelism when multiple solves are requested

## MiniZinc Solver Configuration

### `mzn.solver`
- **`backend`** (string, default: `"coinbc"`)
  - MiniZinc solver backend
  - Options: `coinbc`, `gecode`, `chuffed`, `or-tools`, etc.
  - Must be installed on your system
  - Check available: `minizinc --solvers`
  - CLI override: `--solver-backend`
  
- **`use_python_bindings`** (boolean, default: `true`)
  - Use minizinc-python bindings if true
  - Falls back to CLI if false or bindings unavailable
  
- **`time_limit`** (integer, default: `60`)
  - Maximum solver time in seconds
  - Can be overridden per-solve via API

### `mzn.mcp_server`
Same as `mathprog.mcp_server` but with different defaults:
- **`http_port`**: `8766` (default)

### `mzn.validation`
- **`strict_mode`** (boolean, default: `false`)
  - Enable stricter validation rules
  - Not yet fully implemented

### `mzn.performance`
- **`max_parallel_solvers`** (integer, default: `5`)
  - Maximum concurrent MiniZinc solver instances

## Modeller-Checker Workflow Configuration

### `modeller_checker.modeller`
Configuration for the Modeller LLM (creates optimization models):

- **`provider`** (string, default: `"ollama"`)
  - LLM provider: `ollama`, `openai`, `anthropic`, `azure`
  
- **`model`** (string, default: `"qwen3"`)
  - Model name (provider-specific)
  - Examples: `qwen3`, `gpt-4`, `claude-3-sonnet-20240229`
  
- **`base_url`** (string, default: `"http://127.0.0.1:11434"`)
  - Base URL for Ollama server (Ollama only)
  
- **`temperature`** (float, default: `0.5`)
  - LLM temperature (0.0 = deterministic, 1.0 = creative)
  
- **`api_key`** (string, optional)
  - API key for cloud providers
  - Supports env var expansion: `"${OPENAI_API_KEY}"`
  
- **`max_tokens`** (integer, optional)
  - Maximum tokens in response
  - Falls back to `llm_defaults.default_max_tokens`
  
- **`azure_endpoint`** (string, optional, Azure only)
  - Azure OpenAI endpoint URL
  
- **`api_version`** (string, default: `"2024-02-15-preview"`, Azure only)
  - Azure OpenAI API version

### `modeller_checker.checker`
Configuration for the Checker LLM (validates and critiques models):
- Same parameters as `modeller`
- Can use a different provider/model than modeller
- Typically uses lower temperature (e.g., `0.3`) for more critical analysis

### `modeller_checker.workflow`
- **`max_iterations`** (integer, default: `5`)
  - Maximum refinement iterations between modeller and checker
  - Prevents infinite loops
  
- **`verbose`** (boolean, default: `false`)
  - Enable verbose logging of workflow steps
  - Shows intermediate models and feedback
  
- **`solver_backend`** (string, default: `"minizinc"`)
  - Solver to use: `minizinc` (LP support planned for future)

### `modeller_checker.mcp_server`
Same as other MCP server configs:
- **`http_port`**: `8767` (default)

### `modeller_checker.llm_defaults`
Default values for LLM parameters when not specified:

- **`default_max_tokens`** (integer, default: `2048`)
  - Default max tokens for Anthropic/Azure models
  
- **`default_temperature`** (float, default: `0.5`)
  - Default temperature (not currently used, may be used in future)

## Global Configuration

### `global`
Settings that apply across all components:

- **`log_level`** (string, default: `"INFO"`)
  - Global logging level
  - Can be overridden per-component
  
- **`log_file`** (string, default: `null`)
  - Path to log file
  - Set to enable file logging
  - Example: `"logs/optimise_mcp.log"`
  
- **`debug_mode`** (boolean, default: `false`)
  - Enable debug mode
  - Not yet fully implemented

## Configuration Loading

### Priority Order
1. **CLI arguments** (highest priority)
2. **Config file** (`config.yaml`)
3. **Default values** (lowest priority)

### Config File Locations
1. Specified via `--config /path/to/config.yaml` (CLI)
2. `{repo_root}/config.yaml` (default)

### Environment Variable Expansion
API keys and other sensitive values support environment variable expansion:

```yaml
modeller_checker:
  modeller:
    api_key: "${OPENAI_API_KEY}"  # Expands to env var value
```

Set environment variables before running:
```bash
export OPENAI_API_KEY=sk-...
python -m src.modeller_checker.mcp --stdio
```

## CLI Override Examples

### LP Solver
```bash
# Override solver backend and port
python -m src.mathprog.mcp --http \
  --solver-backend highs \
  --http-port 9000 \
  --log-level DEBUG

# Use custom config file
python -m src.mathprog.mcp --stdio --config /path/to/my_config.yaml
```

### MiniZinc Solver
```bash
# Override solver backend
python -m src.mzn.mcp --http \
  --solver-backend gecode \
  --http-port 9001

# Change host binding
python -m src.mzn.mcp --http \
  --http-host 0.0.0.0 \
  --http-port 8766
```

### Modeller-Checker
```bash
# Override max iterations (CLI script)
python scripts/langchain_modeller_checker.py \
  --iterations 10 \
  --config /path/to/config.yaml \
  --verbose

# MCP server with custom config
python -m src.modeller_checker.mcp --stdio \
  --config /path/to/config.yaml
```

## Example Configurations

### Local Development (Ollama)
```yaml
modeller_checker:
  modeller:
    provider: "ollama"
    model: "qwen3"
    base_url: "http://127.0.0.1:11434"
    temperature: 0.5
  
  checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
```

### Cloud (OpenAI + Anthropic)
```yaml
modeller_checker:
  modeller:
    provider: "openai"
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.5
    max_tokens: 2000
  
  checker:
    provider: "anthropic"
    model: "claude-3-sonnet-20240229"
    api_key: "${ANTHROPIC_API_KEY}"
    temperature: 0.3
    max_tokens: 1500
```

### Azure OpenAI
```yaml
modeller_checker:
  modeller:
    provider: "azure"
    model: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    azure_endpoint: "https://your-resource.openai.azure.com/"
    api_version: "2024-02-15-preview"
    temperature: 0.5
```

### High-Performance MiniZinc
```yaml
minizinc:
  solver:
    backend: "gecode"
    time_limit: 300  # 5 minutes
  
  performance:
    max_parallel_solvers: 10  # More concurrent solvers
  
  mcp_server:
    http_port: 8766
    log_level: "DEBUG"
```

## Validation

After modifying `config.yaml`, validate it loads correctly:

```bash
# Test LP server
python -m src.mathprog.mcp --stdio
# Should show: "Loaded config from .../config.yaml"

# Test MiniZinc server
python -m src.mzn.mcp --stdio

# Test modeller-checker
python scripts/langchain_modeller_checker.py --verbose
```

## See Also

- [README.md](../README.md) - Main documentation
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration patterns
- [MODELLER_CHECKER.md](MODELLER_CHECKER.md) - Workflow details
