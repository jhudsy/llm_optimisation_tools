# Complex Workflow Quick Reference

## Quick Start

```bash
# Test the complex workflow
python scripts/complex_workflow_test.py -v \
  -p "Optimization problem description here..."

# Or use the simple workflow
python scripts/langchain_modeller_checker.py -v \
  -p "Optimization problem description here..."
```

## Key Differences

| Aspect | Simple (2-Agent) | Complex (5-Agent) |
|--------|------------------|-------------------|
| **Agents** | 2 (Modeller, Checker) | 5 (Formulator, Eq. Checker, Translator, Code Checker, Solver Executor) |
| **Phases** | Model → Validate → Solve | Formulate → Validate → Translate → Validate → Solve |
| **Output** | MiniZinc + Solution | Math Equations + MiniZinc + Solution |
| **Error Routing** | All to Modeller | Intelligent routing per error type |
| **Iterations** | 3-5 typical | 5-10 typical |
| **Best For** | Simple problems | Complex formulations |

## Agent Roles

### Simple Workflow
1. **Modeller**: Problem → MiniZinc code
2. **Checker**: Validates MiniZinc correctness

### Complex Workflow
1. **Formulator**: Problem → Mathematical equations
2. **Equation Checker**: Validates equations match problem
3. **Translator**: Equations → MiniZinc code
4. **Code Checker**: Validates MiniZinc implementation
5. **Solver Executor**: Runs solver, diagnoses errors

## Error Routing in Complex Workflow

```
┌─────────────────────┬──────────────────┬─────────────────────┐
│ Error Source        │ Error Type       │ Routes To           │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Equation Checker    │ Math issues      │ → Formulator        │
│ Syntax Validator    │ MiniZinc syntax  │ → Translator        │
│ Code Checker        │ Implementation   │ → Translator        │
│ Solver Executor     │ MiniZinc error   │ → Translator        │
│ Solver Executor     │ Math error       │ → Formulator        │
└─────────────────────┴──────────────────┴─────────────────────┘
```

## Configuration

### Enable Complex Workflow

In `config.yaml`:
```yaml
modeller_checker:
  workflow:
    mode: "complex"  # or "simple"
    max_iterations: 10
```

### Agent-Specific Models

```yaml
modeller_checker:
  formulator:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.5
  
  equation_checker:
    provider: "anthropic"
    model: "claude-3-sonnet"
    temperature: 0.3
  
  # ... etc for other agents
```

## Python API

### LangChain Tool - Simple
```python
from langchain_optimise.modeller_checker_tool import create_modeller_checker_tool

tool = create_modeller_checker_tool(verbose=True)
result = tool.invoke({
    "problem": "Maximize profit...",
    "max_iterations": 10
})
```

### LangChain Tool - Complex
```python
from langchain_optimise.complex_workflow_tool import create_complex_workflow_tool

tool = create_complex_workflow_tool(verbose=True)
result = tool.invoke({
    "problem": "Maximize profit...",
    "max_iterations": 10
})
```

## MCP Server

Both workflows exposed as separate tools:

```bash
# Start server
python -m src.modeller_checker.mcp --stdio
# or
python -m src.modeller_checker.mcp --http --http-port 8767
```

### Available MCP Tools

1. **`modeller_checker_workflow`**
   - Simple 2-agent workflow
   - Fast, straightforward
   
2. **`complex_workflow`**
   - Full 5-agent workflow
   - Detailed formulation
   - Intelligent error routing

## Output Structure

### Simple Workflow
```python
{
    "success": bool,
    "checker_approval": bool,
    "iterations": int,
    "mzn_code": str,
    "final_response": str
}
```

### Complex Workflow
```python
{
    "success": bool,
    "iterations": int,
    "workflow_trace": list[str],  # ["Iter1:Formulator", "Iter1:EquationChecker", ...]
    "formulation": dict,          # Variables, constraints, objective
    "mzn_code": str,
    "solution": dict,             # {var_name: value}
    "final_response": str
}
```

## Example Problem

```python
problem = """
We have 110 acres of land. We can plant wheat or corn.
Wheat yields $40 profit per acre and requires 3 labour hours per acre.
Corn yields $30 profit per acre and requires 2 labour hours per acre.
We have 240 labour hours available.
How should we allocate the land to maximize profit?
"""
```

### Expected Output (Complex)

**Mathematical Formulation:**
```
Variables:
  wheat: 0 ≤ wheat ≤ 110 (acres)
  corn: 0 ≤ corn ≤ 110 (acres)

Constraints:
  wheat + corn ≤ 110  (land)
  3*wheat + 2*corn ≤ 240  (labour)

Objective: maximize 40*wheat + 30*corn
```

**MiniZinc Code:**
```minizinc
var 0..110: wheat;
var 0..110: corn;
constraint wheat + corn <= 110;
constraint 3*wheat + 2*corn <= 240;
solve maximize 40*wheat + 30*corn;
```

**Solution:**
```
wheat = 60
corn = 30
profit = $3300
```

## Debugging

### Enable Verbose Mode
```bash
python scripts/complex_workflow_test.py -v
```

### Check Workflow Trace
The `workflow_trace` field shows exactly which agents ran:
```python
"workflow_trace": [
    "Iter1:Formulator",
    "Iter1:EquationChecker",
    "Iter2:Translator",
    "Iter2:CodeChecker", 
    "Iter2:Solver"
]
```

### Debug Logging
```bash
python -m src.modeller_checker.mcp --stdio --debug
```

## Common Issues

### Q: Workflow not converging?
**A:** Increase `max_iterations` in config or CLI:
```bash
python scripts/complex_workflow_test.py -i 15
```

### Q: Wrong agent receiving errors?
**A:** Check Solver Executor error classification in logs with `--debug`

### Q: Want faster iteration?
**A:** Use simple workflow instead:
```bash
python scripts/langchain_modeller_checker.py
```

### Q: Different models for different agents?
**A:** Configure each agent separately in `config.yaml`

## Files Reference

| File | Purpose |
|------|---------|
| `src/modeller_checker/workflow.py` | Simple 2-agent workflow |
| `src/modeller_checker/complex_workflow.py` | Complex 5-agent workflow |
| `src/modeller_checker/mcp.py` | MCP server (both workflows) |
| `src/langchain_optimise/modeller_checker_tool.py` | LangChain tool (simple) |
| `src/langchain_optimise/complex_workflow_tool.py` | LangChain tool (complex) |
| `scripts/langchain_modeller_checker.py` | CLI (simple) |
| `scripts/complex_workflow_test.py` | CLI (complex) |
| `config.yaml` | Configuration |
| `docs/COMPLEX_WORKFLOW.md` | Full documentation |
| `docs/WORKFLOW_DIAGRAMS.md` | Visual comparison |

## See Also

- [Full Documentation](COMPLEX_WORKFLOW.md)
- [Workflow Diagrams](WORKFLOW_DIAGRAMS.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Main README](../README.md)
