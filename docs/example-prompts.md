# Example Prompts for MCP Clients

Ready-made prompts for driving optimise-mcp from MCP-compatible clients (Ollama, Claude Desktop, etc.).

## LP/MILP Solver Prompts

### 1. Farm Planning Problem
```
I have 110 acres of land. I can plant wheat or corn.
Wheat yields $40 profit per acre and requires 3 labour hours per acre.
Corn yields $30 profit per acre and requires 2 labour hours per acre.
I have 240 labour hours available.
How should I allocate the land to maximize profit?

Please create an LP model and solve it.
```

### 2. Diet Optimization
```
Create a diet optimization model that minimizes cost while meeting nutritional requirements:
- Foods: Rice ($2/kg, 300 cal/kg), Beans ($3/kg, 350 cal/kg), Chicken ($8/kg, 250 cal/kg)
- Requirements: At least 2000 calories, at least 50g protein, at most 70g fat
- Use the LP solver to find the optimal diet.
```

### 3. Production Planning MILP
```
A factory can produce products A and B.
Product A: $50 profit, 2 hours machine time, 1 hour labor
Product B: $40 profit, 1 hour machine time, 2 hours labor
We have 100 hours of machine time and 80 hours of labor available.
Both products must be produced in whole units (integers).
Maximize profit using the LP solver.
```

## MiniZinc Solver Prompts

### 1. Scheduling Problem
```
We need to schedule 5 tasks with durations [3, 2, 4, 1, 3] hours.
Task dependencies: task 2 must finish before task 4, task 1 before task 3.
We want to minimize the total completion time (makespan).
Create and solve a MiniZinc model for this.
```

### 2. Resource Allocation
```
Assign 4 workers to 4 tasks. Each worker has different efficiency for each task:
Worker 1: [9, 2, 7, 8]
Worker 2: [6, 4, 3, 7]
Worker 3: [5, 8, 1, 8]
Worker 4: [7, 6, 9, 4]
Each worker does exactly one task. Maximize total efficiency using MiniZinc.
```

## Modeller-Checker Workflow Prompts

### 1. Basic Optimization
```
Use the modeller-checker workflow to solve this problem:
We have 110 acres of land and can plant wheat or corn.
Wheat: $40 profit/acre, 3 hours labor/acre
Corn: $30 profit/acre, 2 hours labor/acre
Labor available: 240 hours
Maximize profit.
```

### 2. Complex Scheduling
```
Use the dual-agent workflow to create an optimized schedule:
We have 5 jobs with processing times [10, 5, 8, 12, 6] minutes.
Job dependencies: job 1 must complete before job 3, job 2 before job 4.
We want to minimize the makespan while respecting dependencies.
Let the modeller create the model and the checker validate it.
```

### 3. Multi-Constraint Problem
```
Ask the modeller-checker to solve:
A delivery company has 3 trucks with capacities [100, 150, 120] kg.
There are 8 packages to deliver with weights [20, 35, 40, 25, 50, 30, 45, 15] kg.
Each package must be assigned to exactly one truck.
Minimize the maximum truck load (balance the loads).
```

## Workflow Tips

### For LP/MILP:
1. State decision variables clearly
2. Define objective (maximize or minimize)
3. List all constraints
4. Specify variable bounds
5. Indicate if variables must be integers

### For MiniZinc:
1. Describe the problem structure
2. Mention any special constraints (all-different, cumulative, etc.)
3. State optimization goal clearly
4. Provide data values explicitly

### For Modeller-Checker:
1. Provide complete problem description
2. Include all constraints and objectives
3. Let the workflow iterate to refine the model
4. Review the final validated model and solution

## Testing Server Availability

```
List the available tools and show me their descriptions.
```

This will confirm the MCP server is running and show which solvers are available.
