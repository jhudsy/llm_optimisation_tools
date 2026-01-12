# Complex 5-Agent Workflow

The complex workflow extends the simple 2-agent workflow with specialized agents for each stage of optimization problem solving.

## Architecture

### Agents

1. **Formulator** - Converts natural language problem into mathematical equations
   - Input: Natural language problem description
   - Output: Mathematical formulation (variables, constraints, objective)
   - Does NOT generate code - only pure mathematics

2. **Equation Checker** - Validates mathematical formulation
   - Input: Problem + Mathematical formulation
   - Output: Approve/Reject with specific feedback
   - Checks completeness, correctness, mathematical validity

3. **Translator** - Converts equations to MiniZinc code
   - Input: Validated mathematical formulation
   - Output: MiniZinc code
   - Handles syntax validation and MiniZinc-specific requirements

4. **Code Checker** - Validates MiniZinc implementation
   - Input: Mathematical formulation + MiniZinc code
   - Output: Approve/Reject with specific feedback
   - Verifies code correctly implements the formulation

5. **Solver Executor** - Runs solver and diagnoses errors
   - Input: MiniZinc code
   - Output: Solution OR error diagnosis with routing
   - Classifies errors and routes feedback to appropriate agent

## Error Routing

The Solver Executor intelligently routes errors:

- **MiniZinc syntax/semantic errors** → Translator
  - Example: Invalid MiniZinc syntax, wrong constraint format
  
- **Mathematical formulation issues** → Formulator
  - Example: Non-linear model for linear solver, infeasible constraints
  
- **Unknown/general errors** → Translator (default)

Errors caught by Equation Checker → Formulator
Errors caught by Code Checker → Translator

## Workflow Phases

The workflow progresses through phases:

```
formulation → equation_check → translation → code_check → solving
     ↑            ↑                 ↑            ↑           ↓
     └────────────┴─────────────────┴────────────┴───(error routing)
```

## Configuration

In `config.yaml`, configure each agent separately:

```yaml
modeller_checker:
  # Configure each agent individually
  formulator:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.5
  
  equation_checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3  # Lower temp for validation
  
  translator:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.4
  
  code_checker:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.2  # Lowest temp for strict checking
  
  solver_executor:
    provider: "ollama"
    model: "qwen3"
    temperature: 0.3
  
  workflow:
    mode: "complex"  # Set to "simple" for 2-agent workflow
    max_iterations: 10
```

You can use different models for each agent, or even different providers (e.g., GPT-4 for formulation, Claude for checking).

## Usage

### CLI Script

```bash
python scripts/complex_workflow_test.py \
  --problem "Your optimization problem..." \
  --verbose \
  --iterations 10
```

### LangChain Tool

```python
from langchain_optimise.complex_workflow_tool import create_complex_workflow_tool

tool = create_complex_workflow_tool(verbose=True)
result = tool.invoke({
    "problem": "We have 110 acres...",
    "max_iterations": 10
})
```

### MCP Server

Both workflows are exposed via MCP:

```python
# Simple 2-agent workflow
modeller_checker_workflow(problem="...", max_iterations=10)

# Complex 5-agent workflow
complex_workflow(problem="...", max_iterations=10)
```

## Benefits Over Simple Workflow

1. **Separation of Concerns**: Each agent has a specific, focused role
2. **Better Error Routing**: Intelligent feedback to the right agent
3. **Mathematical Validation**: Explicit formulation step before coding
4. **Clearer Debugging**: Workflow trace shows which agents were involved
5. **Flexibility**: Can swap different models for different roles

## Output

The complex workflow provides:
- Workflow trace (which agents were invoked)
- Mathematical formulation (variables, constraints, objective)
- MiniZinc code
- Optimal solution
- Success/failure status

Example output:
```
Success: True
Iterations: 5
Workflow Trace: Iter1:Formulator -> Iter1:EquationChecker -> Iter2:Translator -> Iter2:CodeChecker -> Iter2:Solver

MATHEMATICAL FORMULATION
========================
DECISION VARIABLES:
  wheat: integer - Acres of wheat to plant
    Bounds: 0 ≤ wheat ≤ 110
  corn: integer - Acres of corn to plant
    Bounds: 0 ≤ corn ≤ 110

CONSTRAINTS:
  1. wheat + corn ≤ 110
     (Total land constraint)
  2. 3*wheat + 2*corn ≤ 240
     (Labour hours constraint)

OBJECTIVE: MAXIMIZE
  40*wheat + 30*corn

MINIZINC MODEL
==============
var 0..110: wheat;
var 0..110: corn;
constraint wheat + corn <= 110;
constraint 3*wheat + 2*corn <= 240;
solve maximize 40*wheat + 30*corn;
```

## When to Use

**Use Complex Workflow when:**
- You need explicit mathematical formulation
- Problem requires careful validation at each stage
- You want detailed workflow tracing
- Different expertise levels for different agents

**Use Simple Workflow when:**
- Problem is straightforward
- Speed is priority
- Don't need mathematical formulation separate from code
