# Multi-Agent Modeller-Checker Workflow

## Overview

This workflow implements a two-agent LangChain system where:

1. **Modeller Agent**: Takes a problem description and creates a MiniZinc optimization model
2. **Checker Agent**: Verifies that the MiniZinc model correctly represents the problem
3. **Feedback Loop**: If the checker finds issues, feedback goes back to the modeller for refinement
4. **Solver**: Once approved, the model is solved and the solution returned

## Architecture

```
Problem Description
        ↓
    [MODELLER]  → Creates MiniZinc Model
        ↓
   [Syntax Validation]
        ↓
    [CHECKER]  → Verifies Model vs Problem
        ↓
    ┌─────────────────────┐
    │ Approved? Yes       │  No
    └─────────────────────┘
         ↓                 ↓
      [SOLVER]        [Feedback to Modeller]
         ↓                 ↓
      Solution       (Loop back to Modeller)
```

## Usage

```bash
# Basic usage (qwen3 by default)
python scripts/langchain_modeller_checker.py -v

# Specify model and temperature
python scripts/langchain_modeller_checker.py -v --model mistral --temperature 0.3

# Set max iterations for modeller-checker loop
python scripts/langchain_modeller_checker.py -v --iterations 5

# Custom problem
python scripts/langchain_modeller_checker.py \
  -v \
  -p "A company has 1000 units of capacity..." \
  --model qwen3 \
  --temperature 0.5
```

## CLI Arguments

- `-p, --problem`: Problem description (default: farm allocation problem)
- `-m, --model`: Ollama model to use (default: `qwen3`)
- `-t, --temperature`: LLM temperature for creativity (default: 0.5)
- `-v, --verbose`: Show detailed workflow steps
- `-i, --iterations`: Max modeller-checker iterations (default: 5)

## How It Works

### Phase 1: Modeller
- Receives problem description
- Creates MiniZinc model with proper syntax:
  - Variable declarations
  - Constraints
  - Objective (`solve maximize` or `solve minimize`)
- Returns JSON with `action: "provide_model"` and `mzn_code`

### Phase 2: Syntax Validation
- Validates MiniZinc syntax
- Reports any syntax errors
- If errors: Feedback loops to modeller

### Phase 3: Checker
- Analyzes model against original problem
- Verifies:
  - All constraints are represented
  - Objective function is correct
  - Variables match the problem
  - Syntax is valid
- Returns JSON with `action: "approve" | "reject"`
- If rejected: Issues list goes back to modeller

### Phase 4: Iteration
- If checker approves: Proceed to solver
- If checker rejects: Send feedback back to modeller (goes to next iteration)
- Max iterations prevent infinite loops

### Phase 5: Solver
- Executes the approved MiniZinc model
- Returns optimal solution

## Example Output

```
================================================================================
MODELLER-CHECKER WORKFLOW
================================================================================
Problem: We have 110 acres of land...
Model: qwen3, Temperature: 0.5

================================================================================
ITERATION 1
================================================================================

[MODELLER PHASE]
Modeller response:
{
    "action": "provide_model",
    "mzn_code": "var int: wheat;\nvar int: corn;\n...",
    "explanation": "This model defines two decision variables..."
}

[SYNTAX VALIDATION]
Validation: {'valid': True, 'issues': []}
✓ Syntax is valid

[CHECKER PHASE]
Checker response:
{
    "action": "approve",
    "reasoning": "The model correctly captures all problem constraints..."
}

✓ CHECKER APPROVED - Proceeding to solve

[SOLVE PHASE]
Solve result: {'solver_name': 'minizinc', 'run_status': 'success', ...}
✓ SUCCESS: Solver found optimal solution

================================================================================
WORKFLOW SUMMARY
================================================================================
Iterations: 1
Checker Approved: True
Success: True
Final Response: Solver found optimal solution: {'wheat': '20', 'corn': '90'}
```

## Feedback Loop Example

When the checker finds issues:

```
[CHECKER PHASE]
Checker response:
{
    "action": "reject",
    "reasoning": "Model doesn't properly represent the constraint...",
    "issues": ["Constraint A is missing", "Objective calculation is wrong"]
}

✗ CHECKER REJECTED
  - Constraint A is missing
  - Objective calculation is wrong

================================================================================
ITERATION 2
================================================================================

[MODELLER PHASE]
Modeller input: The Checker found issues with the MiniZinc model:
  - Constraint A is missing
  - Objective calculation is wrong

Please fix these issues and provide an improved model.
```

Then the modeller refines the model and the cycle repeats.

## Key Design Features

1. **Separation of Concerns**: Modeller handles creation, checker handles verification
2. **Feedback Loop**: Modeller learns from checker's feedback
3. **Syntax Validation**: Early detection of MiniZinc syntax errors
4. **Iterative Refinement**: Multiple rounds if needed
5. **Semantic Checking**: Checker verifies logic matches problem, not just syntax
6. **Deterministic Solving**: Once approved, model is definitively solved

## Implementation Details

### Modeller System Prompt
- Emphasized proper MiniZinc syntax (`solve maximize` vs `maximize`)
- Clear variable declaration rules
- Complete, runnable code requirement

### Checker System Prompt
- Semantic analysis of problem vs model
- Syntax verification
- Issue specificity for feedback

### Tools Used
- `create_validate_minizinc_tool()`: Syntax validation
- `create_solve_minizinc_tool()`: Solver execution
- `ChatOllama`: LLM backend via Ollama

## Testing Results

### qwen3 Model
- ✓ Successful completion
- ✓ Correct MiniZinc generation
- ✓ Proper syntax
- ✓ Optimal solution found (wheat=20, corn=90)

### mistral Model
- ⚠️ Workflow completes but generates suboptimal syntax
- Note: Validation tool may need stricter regex patterns

## Future Enhancements

1. **Stricter Validation**: Improve MiniZinc regex validation
2. **Multi-turn Refinement**: More sophisticated feedback without restarting
3. **Constraint Analysis**: Separate validation for constraints vs objective
4. **Explanation Requests**: Modeller explains reasoning for design choices
5. **Consistency Checks**: Verify variable domains match constraints
6. **Timeout Handling**: Graceful handling of long solver runs
7. **Solution Quality**: Check if solution is reasonable before accepting

## File Location

`scripts/langchain_modeller_checker.py` (418 lines)
