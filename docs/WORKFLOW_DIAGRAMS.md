# Workflow Comparison

## Simple 2-Agent Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Problem Description                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   MODELLER AGENT     │
              │  Problem → MiniZinc  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Syntax Validator   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   CHECKER AGENT      │
              │ Validate correctness │
              └──────────┬───────────┘
                         │
                    ┌────┴────┐
                    │         │
                Approve    Reject
                    │         │
                    │         └─────┐
                    │               │
                    ▼               ▼
           ┌──────────────┐  Feedback to
           │    SOLVER    │   Modeller
           └──────┬───────┘       │
                  │               │
             ┌────┴────┐          │
             │         │          │
          Success   Error         │
             │         └──────────┘
             ▼
        ┌─────────┐
        │Solution │
        └─────────┘

Iterations: Typically 3-5
Phases: 2 (Modeling → Checking)
```

## Complex 5-Agent Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Problem Description                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  FORMULATOR AGENT    │
              │ Problem → Equations  │
              │  (Pure Mathematics)  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ EQUATION CHECKER     │
              │ Validate formulation │
              └──────────┬───────────┘
                         │
                    ┌────┴────┐
                    │         │
                Approve    Reject
                    │         │
                    │         └──────────┐
                    │                    │
                    ▼                    ▼
           ┌──────────────────┐  Feedback to
           │ TRANSLATOR AGENT │   Formulator
           │ Equations → Mzn  │         │
           └────────┬─────────┘         │
                    │                   │
                    ▼                   │
           ┌──────────────────┐         │
           │ Syntax Validator │         │
           └────────┬─────────┘         │
                    │                   │
                    ▼                   │
           ┌──────────────────┐         │
           │  CODE CHECKER    │         │
           │ Validate MiniZinc│         │
           └────────┬─────────┘         │
                    │                   │
               ┌────┴────┐              │
               │         │              │
           Approve    Reject            │
               │         │              │
               │         └──────────┐   │
               │                    │   │
               ▼                    ▼   │
      ┌──────────────┐      Feedback to│
      │    SOLVER    │       Translator│
      └──────┬───────┘              │   │
             │                      │   │
        ┌────┴────┐                 │   │
        │         │                 │   │
     Success   Error                │   │
        │         │                 │   │
        │         ▼                 │   │
        │  ┌────────────────┐       │   │
        │  │ SOLVER EXECUTOR│       │   │
        │  │ Error Diagnosis│       │   │
        │  └────────┬───────┘       │   │
        │           │               │   │
        │      ┌────┴────┬──────────┘   │
        │      │         │              │
        │   MiniZinc  Mathematical      │
        │    Error     Error            │
        │      │         │              │
        │      └─────────┴──────────────┘
        │
        ▼
   ┌─────────┐
   │Solution │
   └─────────┘

Iterations: Typically 5-10
Phases: 5 (Formulation → Eq. Check → Translation → Code Check → Solving)
```

## Error Routing Comparison

### Simple Workflow
```
Modeller Error → Modeller (retry)
Checker Reject → Modeller (revise)
Solver Error → Modeller (fix)
```

### Complex Workflow
```
Formulator Error → Formulator (retry)
Equation Reject → Formulator (revise)
Translator Error → Translator (retry)
Code Reject → Translator (revise)
Solver MiniZinc Error → Translator (fix code)
Solver Math Error → Formulator (fix formulation)
```

## Data Flow

### Simple Workflow
```
Problem → MiniZinc Code → Validation → Solution
```

### Complex Workflow
```
Problem → Math Equations → Validation → 
MiniZinc Code → Validation → Solution

With explicit intermediate representation!
```

## Agent Responsibilities

### Simple (2 agents)
| Agent | Input | Output | Temperature |
|-------|-------|--------|-------------|
| Modeller | Problem | MiniZinc | 0.5 |
| Checker | Problem + Code | Approve/Reject | 0.3 |

### Complex (5 agents)
| Agent | Input | Output | Temperature |
|-------|-------|--------|-------------|
| Formulator | Problem | Equations | 0.5 |
| Eq. Checker | Problem + Equations | Approve/Reject | 0.3 |
| Translator | Equations | MiniZinc | 0.4 |
| Code Checker | Equations + Code | Approve/Reject | 0.2 |
| Solver Executor | Code | Solution/Error | 0.3 |

## When to Use

### Use Simple When:
- ✓ Problem is straightforward
- ✓ Don't need mathematical formulation
- ✓ Speed is important
- ✓ 3-5 iterations acceptable

### Use Complex When:
- ✓ Need explicit formulation
- ✓ Complex mathematical constraints
- ✓ Want detailed workflow trace
- ✓ Different models for different tasks
- ✓ More thorough validation needed
