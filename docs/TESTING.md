# Testing Guide

## Roundtrip Testing Scripts

The `scripts/` directory contains end-to-end tests for LP and MiniZinc solvers using LLMs (via Ollama).

### LangChain Roundtrip Tests (Recommended for Development)

Fast, direct testing without MCP server overhead.

#### LP Test
```bash
python scripts/langchain_lp_roundtrip.py -v
```

**Options:**
- `--model MODEL` – Ollama model (default: `mistral`)
- `--temperature T` – LLM sampling temperature (default: `0.7`)
- `-v, --verbose` – Show each dialogue turn

**What it tests:**
- LP tool creation and initialization
- LangChain + Ollama integration
- Dialogue loop: LLM generates LP → validate → solve
- Error handling and recovery

#### MiniZinc Test
```bash
python scripts/langchain_minizinc_roundtrip.py -v
```

Same options as LP test. Tests MiniZinc constraint programming instead.

### MCP Roundtrip Tests (Full Integration)

Complete MCP protocol testing via stdio or HTTP.

#### LP via MCP
```bash
python scripts/mcp_lp_roundtrip.py --model mistral --transport stdio
```

#### MiniZinc via MCP
```bash
python scripts/mcp_minizinc_roundtrip.py --model mistral --transport stdio
```

**Transport options:** `stdio` (default) or `http`

**What it tests:**
- MCP server startup and shutdown
- Tool schema discovery via MCP
- Multi-turn LLM dialogue through MCP protocol
- HTTP or stdio transport correctness

### Comparison

| Script | Purpose | Startup | Debug | Best For |
|--------|---------|---------|-------|----------|
| `langchain_lp_roundtrip.py` | Direct LP tool test | Fast | Easy | Development |
| `mcp_lp_roundtrip.py` | MCP LP integration | Slow | Hard | Production validation |
| `langchain_minizinc_roundtrip.py` | Direct MiniZinc test | Fast | Easy | Development |
| `mcp_minizinc_roundtrip.py` | MCP MiniZinc integration | Slow | Hard | Production validation |

## Setup

1. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```

2. **Pull a model (if needed):**
   ```bash
   ollama pull mistral
   ```

3. **Install project:**
   ```bash
   pip install -e .
   ```

4. **Run a test:**
   ```bash
   python scripts/langchain_lp_roundtrip.py -v
   ```

## Troubleshooting

**Module not found:**
```bash
cd /path/to/optimise_mcp
pip install -e .
```

**Cannot connect to Ollama:**
```bash
ollama serve  # in another terminal
```

**LLM fails to generate JSON:**
Try a different model or lower temperature:
```bash
python scripts/langchain_lp_roundtrip.py --model neural-chat --temperature 0.3
```

**Tool execution errors:**
Use `-v` flag to see full error details and tool invocations.

## Test Problem

All roundtrip tests solve the same problem to ensure consistency:

**Problem:** 110 acres can grow wheat or corn
- Wheat: $40/acre profit, 3 labour hours/acre
- Corn: $30/acre profit, 2 labour hours/acre
- Constraint: 240 total labour hours

**Expected answer:** wheat=50 acres, corn=30 acres, profit=$2900

## Running Full Test Suite

```bash
pytest tests/          # All unit tests
pytest tests/ -v       # Verbose output
```

Tests use mocked solvers by default. To test with real solvers:
```bash
pytest tests/ --live-minizinc --live-lp
```
