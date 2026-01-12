# Migration Guide: 2-Agent to 5-Agent Workflow

This guide helps you migrate from the deprecated 2-agent workflow to the new 5-agent workflow.

## What Changed

The simple 2-agent workflow (Modeller + Checker) has been replaced with a more sophisticated 5-agent workflow:

**Old (2-agent):**
- Modeller → Checker → Solver

**New (5-agent):**
- Formulator → Equation Checker → Translator → Code Checker → Solver Executor

## Configuration Changes

### Old config.yaml
```yaml
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
```

### New config.yaml
```yaml
modeller_checker:
  formulator:          # Replaces modeller
    provider: "ollama"
    model: "qwen3"
    temperature: 0.5
  equation_checker:    # First validation stage
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  translator:          # Converts equations to code
    provider: "ollama"
    model: "qwen3"
    temperature: 0.4
  code_checker:        # Second validation stage
    provider: "ollama"
    model: "qwen3"
    temperature: 0.2
  solver_executor:     # Runs solver and diagnoses errors
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  workflow:
    max_iterations: 10  # Increased for more thorough processing
```

**Simplest migration:** Just add the 3 new agents (equation_checker, translator, solver_executor). The config loader will use `modeller` and `checker` as fallbacks if needed.

## Code Changes

### MCP Server
**Old:**
```python
# Two tools exposed
modeller_checker_workflow(problem="...", max_iterations=5)  # Simple
complex_workflow(problem="...", max_iterations=10)         # Complex
```

**New:**
```python
# Single tool
optimization_workflow(problem="...", max_iterations=10)
```

### LangChain Tool
**Old:**
```python
# Simple workflow
from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool
tool = create_modeller_checker_tool()

# Complex workflow
from langchain_optimise.complex_workflow_tool import create_complex_workflow_tool
tool = create_complex_workflow_tool()
```

**New:**
```python
from langchain_optimise.workflow_tool import create_optimization_workflow_tool
tool = create_optimization_workflow_tool()
```

### CLI Scripts
**Old:**
```bash
# Simple
python scripts/langchain_modeller_checker.py -v -p "problem..."

# Complex
python scripts/complex_workflow_test.py -v -p "problem..."
```

**New:**
```bash
python scripts/workflow_test.py -v -p "problem..."
```

### Python API
**Old:**
```python
from modeller_checker.workflow import run_modeller_checker_workflow
from modeller_checker.complex_workflow import run_complex_workflow

# Had to choose which workflow to use
result = await run_complex_workflow(...)
```

**New:**
```python
from modeller_checker.workflow import run_workflow

# Single workflow function
result = await run_workflow(...)
```

## Function Renames

| Old | New |
|-----|-----|
| `run_modeller_checker_workflow()` | `run_workflow()` |
| `run_complex_workflow()` | `run_workflow()` |
| `create_modeller_checker_tool()` | `create_optimization_workflow_tool()` |
| `create_complex_workflow_tool()` | `create_optimization_workflow_tool()` |
| `create_complex_workflow_llms()` | `create_llms_from_config()` |

## File Locations

| Old | New |
|-----|-----|
| `src/modeller_checker/workflow.py` (simple) | Archived as `workflow_simple_deprecated.py` |
| `src/modeller_checker/complex_workflow.py` | Renamed to `workflow.py` |
| `src/langchain_optimise/modeller_checker_tool.py` | Archived as `modeller_checker_tool_deprecated.py` |
| `src/langchain_optimise/complex_workflow_tool.py` | Renamed to `workflow_tool.py` |
| `scripts/langchain_modeller_checker.py` | Archived as `langchain_modeller_checker_deprecated.py` |
| `scripts/complex_workflow_test.py` | Renamed to `workflow_test.py` |

## Key Improvements

### 1. Explicit Mathematical Formulation
The new workflow creates an explicit mathematical representation before generating code:
```
Problem → Equations (validated) → MiniZinc (validated) → Solution
```

### 2. Intelligent Error Routing
Errors are classified and routed intelligently:
- **MiniZinc errors** → Translator (fix code)
- **Math errors** (infeasible, non-linear) → Formulator (fix equations)
- **Validation errors** → Appropriate checker

### 3. Workflow Tracing
See exactly what happened:
```python
{
    "workflow_trace": [
        "Iter1:Formulator",
        "Iter1:EquationChecker",
        "Iter2:Translator",
        "Iter2:CodeChecker",
        "Iter2:Solver"
    ]
}
```

### 4. More Flexible Configuration
Each agent can use:
- Different models
- Different providers
- Different temperatures

Example - Mix providers:
```yaml
formulator:
  provider: "openai"
  model: "gpt-4"       # Use GPT-4 for creative formulation
equation_checker:
  provider: "anthropic"
  model: "claude-3-sonnet"  # Use Claude for rigorous checking
translator:
  provider: "ollama"
  model: "qwen3"       # Use local model for code generation
```

## Output Format Changes

### Old (2-agent)
```python
{
    "success": bool,
    "checker_approval": bool,
    "iterations": int,
    "mzn_code": str,
    "final_response": str
}
```

### New (5-agent)
```python
{
    "success": bool,
    "iterations": int,
    "workflow_trace": list[str],  # NEW: detailed trace
    "formulation": dict,           # NEW: mathematical formulation
    "mzn_code": str,
    "solution": dict,              # NEW: parsed solution variables
    "final_response": str
}
```

## Backwards Compatibility

### Deprecated Files Still Available
Old files are preserved with `_deprecated` suffix:
- `src/modeller_checker/workflow_simple_deprecated.py`
- `src/langchain_optimise/modeller_checker_tool_deprecated.py`
- `scripts/langchain_modeller_checker_deprecated.py`

### Config Fallbacks
The new config loader will use old `modeller`/`checker` configs as fallbacks:
```python
# If formulator not configured, uses modeller
# If equation_checker not configured, uses checker
# etc.
```

So old configs will still work (but with warnings).

## Migration Checklist

- [ ] Update `config.yaml` to include all 5 agents
- [ ] Change MCP tool calls from `modeller_checker_workflow` to `optimization_workflow`
- [ ] Update LangChain imports from `modeller_checker_tool` to `workflow_tool`
- [ ] Update function calls: `create_modeller_checker_tool()` → `create_optimization_workflow_tool()`
- [ ] Update CLI scripts to use `scripts/workflow_test.py`
- [ ] Update Python imports: `from modeller_checker.workflow import run_workflow`
- [ ] Increase `max_iterations` from 5 to 10 (recommended)
- [ ] Update any documentation/notebooks to reference new workflow

## Benefits

### Why Migrate?

1. **Better Error Handling**: Errors routed to the right agent
2. **Explicit Formulation**: Mathematical representation before code
3. **More Thorough**: Multiple validation stages
4. **Better Debugging**: Workflow trace shows exactly what happened
5. **More Flexible**: Configure each agent independently
6. **Future-Proof**: This is the maintained workflow going forward

### Performance Impact

- **More iterations**: Typically 5-10 instead of 3-5
- **More thorough**: Multiple validation stages catch more errors
- **Better results**: Explicit formulation leads to better models

## Troubleshooting

### "max_iterations too low"
**Problem:** Workflow not converging
**Solution:** Increase from 5 to 10 or 15:
```yaml
workflow:
  max_iterations: 15
```

### "Missing agent config"
**Problem:** `KeyError: 'formulator'`
**Solution:** Add all 5 agents to config, or ensure `modeller`/`checker` are present as fallbacks

### "Different tool name in MCP"
**Problem:** `modeller_checker_workflow` not found
**Solution:** Use `optimization_workflow` instead

## Questions?

See the full documentation:
- [docs/WORKFLOW.md](WORKFLOW.md) - Complete workflow guide
- [docs/QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference
- [README.md](../README.md) - Main README with examples
