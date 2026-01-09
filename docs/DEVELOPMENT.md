# Development Guide

## Architecture Overview

optimise-mcp consists of two independent constraint solving systems:

### LP/MILP System
- **`src/mathprog/solver.py`** – HiGHS solver backend with concurrency guard (max 5)
- **`src/mathprog/validator.py`** – LP format validation with quadratic block support
- **`src/mathprog/mcp.py`** – MCP server exposing `validate_lp` and `solve_lp`

### MiniZinc System
- **`src/mzn/solver.py`** – Python API wrapper with CLI fallback
- **`src/mzn/validator.py`** – Syntax error detection with prescriptive messages
- **`src/mzn/mcp.py`** – MCP server exposing `validate_minizinc` and `solve_minizinc`

### LangChain Tools
- **`src/langchain_optimise/lp_tools.py`** – LangChain-compatible LP tool factories
- **`src/langchain_optimise/minizinc_tools.py`** – LangChain-compatible MiniZinc tool factories

## Adding Features

### Add a new validation rule (MiniZinc)

Edit `src/mzn/validator.py` in the `validate_minizinc_text()` function:

```python
# Check for your specific pattern
for idx, raw_line in enumerate(lines, start=1):
    line = raw_line.split("%")[0].strip()  # Skip comments
    if re.search(r'your_pattern', line):
        issues.append(
            MiniZincValidationIssue(
                line_number=idx,
                message="Your prescriptive error message with example",
                line_content=raw_line,
            )
        )
```

Run validation tests:
```bash
python -c "
from src.mzn.validator import validate_minizinc_text
issues = validate_minizinc_text('test code')
for i in issues:
    print(f'Line {i.line_number}: {i.message}')
"
```

### Improve system prompts

Edit the `SYSTEM_PROMPT` in roundtrip scripts:

1. **LangChain roundtrip:** `scripts/langchain_minizinc_roundtrip.py`
2. **MCP roundtrip:** `scripts/mcp_minizinc_roundtrip.py`

Include:
- Explicit syntax rules with correct/incorrect examples
- Complete working example of the problem domain
- Step-by-step workflow

### Handle new solver errors

For MiniZinc in `src/mzn/solver.py`:
- CLI errors are parsed from stderr in the fallback path
- Python API errors are caught and wrapped with context

For LP in `src/mathprog/solver.py`:
- HiGHS status codes are mapped to meaningful strings
- Quota errors trigger "solver resource temporarily unavailable"

## Running Tests

All tests are in `tests/`:
```bash
pytest tests/                 # All tests
pytest tests/test_minizinc_tools.py -v  # MiniZinc tests only
```

Tests use mocked solvers by default. To test with real solvers:
```bash
pytest tests/ --live-minizinc --live-lp
```

## JSON Parsing in Roundtrips

The `extract_json_plans()` function in `scripts/langchain_minizinc_roundtrip.py` handles:

1. **Standard JSON** – direct parsing
2. **Literal newlines** – converts `\n` line breaks to escaped `\\n`
3. **Python string concatenation** – removes `" + "` operators from long strings

If adding new LLM formatting quirks, update the 3-pass recovery logic in `extract_json_plans()`.

## Debugging

### MiniZinc solver issues
Check the CLI execution path in `src/mzn/solver.py`:
```python
# Temp .mzn written to /tmp/tmp*.mzn
# Run: minizinc --solver coinbc /tmp/tmp*.mzn
# Parse stdout for: name=value pairs
```

### LP solver quota
The `ResourceGaurd` in `src/mathprog/solver.py` limits to 5 concurrent instances. Excess calls return:
```json
{"error": "solver resource temporarily unavailable"}
```

### JSON extraction failures
Enable debug output in roundtrip scripts:
```bash
python scripts/langchain_minizinc_roundtrip.py --verbose 2>&1 | grep "DEBUG"
```

## Roundtrip Testing

Two harnesses test LLM integration:

1. **LP:** `scripts/mcp_lp_roundtrip.py`
   - Asks: "Maximize 40*wheat + 30*corn subject to land and labour constraints"
   - Expected: wheat=50, corn=30, profit=2900

2. **MiniZinc:** `scripts/langchain_minizinc_roundtrip.py`
   - Same problem in MiniZinc constraint syntax
   - Tests validation → solving → error recovery loop

Run with Ollama:
```bash
ollama serve &
python scripts/mcp_lp_roundtrip.py --model mistral --transport stdio
python scripts/langchain_minizinc_roundtrip.py
```

## Project Conventions

- **Validators** return `Issue` dataclasses with `line_number`, `message`, `line_content`
- **Solvers** return dicts with `solver_name`, `solver_backend`, `run_status`, `objective_value`, `variables`, `summary`
- **MCP tools** receive structured schema inputs (Pydantic `BaseModel`)
- **Errors** are caught and wrapped with context; never lose the original message

## File Organization

Avoid polluting root directory:
- `.md` files in root: Only `README.md`, core docs in `docs/`
- Examples in `examples/`
- Scripts in `scripts/` with detailed README_ROUNDTRIP.md
- Tests in `tests/` with clear naming: `test_<module>_<feature>.py`

## Next Steps

1. **Better MiniZinc generation** – add more validation rules and prompt patterns
2. **Interactive debugging** – return validation issues in a format agents can use to iterate
3. **Parallel solving** – async solver calls for agent-driven multi-model optimization
4. **Cost tracking** – measure solver time/resources per call
