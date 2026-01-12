"""
Complex multi-agent workflow for optimization modeling.

5-agent workflow:
1. Formulator: Problem → Mathematical equations/constraints
2. Equation Checker: Validates equations match problem
3. Translator: Equations → MiniZinc code
4. Code Checker: Validates MiniZinc code
5. Solver Executor: Runs solver, diagnoses errors, routes feedback
"""

import sys
from pathlib import Path
import json
import re
import logging
from typing import Optional, Any, Literal
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel

# Add src to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

LOGGER = logging.getLogger("modeller_checker.complex_workflow")


# ============================================================================
# AGENT SYSTEM PROMPTS
# ============================================================================

FORMULATOR_SYSTEM_PROMPT = """You are an expert mathematical modeler. Your role is to:
1. Take a natural language optimization problem description
2. Extract decision variables, constraints, and objectives
3. Formulate them as clear mathematical equations/inequalities
4. Use standard mathematical notation (not code)

You should NOT generate MiniZinc or any programming code. Focus on pure mathematics.

Output format (JSON):
{
    "action": "provide_formulation",
    "variables": {
        "variable_name": {"type": "integer|real|boolean", "description": "...", "bounds": "..."}
    },
    "constraints": [
        {"equation": "mathematical constraint", "description": "what it models"}
    ],
    "objective": {
        "type": "maximize|minimize",
        "expression": "mathematical expression"
    },
    "explanation": "Brief summary of the formulation"
}

When receiving feedback about your formulation:
- Address the specific issues raised
- Keep correct parts unchanged
- Provide improved mathematical formulation"""


EQUATION_CHECKER_SYSTEM_PROMPT = """You are an expert at validating mathematical formulations. Your role is to:
1. Review the mathematical formulation against the original problem
2. Check that all constraints from the problem are captured
3. Verify the objective function matches the problem goal
4. Identify any missing, incorrect, or unnecessary constraints
5. Check for mathematical correctness (linearity, feasibility, etc.)

Output format (JSON):
{
    "action": "approve" | "reject",
    "reasoning": "detailed analysis",
    "issues": ["issue1", "issue2", ...] (only if rejecting)
}

Be thorough: check completeness, correctness, and mathematical validity."""


TRANSLATOR_SYSTEM_PROMPT = """You are an expert MiniZinc programmer. Your role is to:
1. Take validated mathematical equations/constraints
2. Translate them into syntactically correct MiniZinc code
3. Use proper MiniZinc syntax and data types
4. Ensure the code is complete and runnable

CRITICAL MiniZinc requirements:
- Variables: 'var type: name;'
- Constraints: 'constraint expression;'
- Objective: 'solve maximize expression;' or 'solve minimize expression;'
- Proper semicolons and syntax

Output format (JSON):
{
    "action": "provide_code",
    "mzn_code": "complete MiniZinc code",
    "explanation": "brief translation notes"
}

When receiving feedback:
- Fix the specific MiniZinc issues raised
- Maintain fidelity to the mathematical formulation
- Provide corrected, complete MiniZinc code"""


CODE_CHECKER_SYSTEM_PROMPT = """You are an expert MiniZinc code reviewer. Your role is to:
1. Review MiniZinc code for syntax correctness
2. Verify it properly implements the mathematical formulation
3. Check for proper use of MiniZinc constructs
4. Identify semantic issues (wrong types, missing constraints, etc.)
5. Verify the solve statement matches the objective

Output format (JSON):
{
    "action": "approve" | "reject",
    "reasoning": "detailed analysis",
    "issues": ["issue1", "issue2", ...] (only if rejecting)
}

Be precise: distinguish between syntax errors and semantic issues."""


SOLVER_EXECUTOR_PROMPT = """You are an expert at diagnosing solver errors. When a solver fails:
1. Analyze the error message
2. Classify the error type:
   - "minizinc_syntax": MiniZinc syntax/semantic errors
   - "mathematical": Mathematical formulation issues (infeasible, unbounded, non-linear for linear solver)
   - "solver_config": Solver configuration or timeout issues
3. Provide specific feedback for the appropriate agent

Output format (JSON):
{
    "error_type": "minizinc_syntax" | "mathematical" | "solver_config",
    "feedback": "specific description of what went wrong",
    "suggestions": ["suggestion1", "suggestion2", ...]
}"""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WorkflowState:
    """Tracks the current state of the workflow."""
    problem: str
    current_formulation: Optional[dict] = None
    current_equations: Optional[str] = None  # Text representation
    current_mzn_code: Optional[str] = None
    iteration: int = 0
    phase: Literal["formulation", "equation_check", "translation", "code_check", "solving"] = "formulation"
    last_feedback: Optional[dict] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_json_from_response(response_text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling various formatting issues."""
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in response
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for match in matches:
        try:
            cleaned = match.replace('\n', ' ').replace('" + "', '')
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    
    return None


def stringify_content(content: Any) -> str:
    """Convert content to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(c) for c in content)
    return str(content)


def format_formulation_for_display(formulation: dict) -> str:
    """Format mathematical formulation as readable text."""
    lines = []
    lines.append("DECISION VARIABLES:")
    for var_name, var_info in formulation.get("variables", {}).items():
        lines.append(f"  {var_name}: {var_info.get('type')} - {var_info.get('description')}")
        if var_info.get('bounds'):
            lines.append(f"    Bounds: {var_info.get('bounds')}")
    
    lines.append("\nCONSTRAINTS:")
    for i, constraint in enumerate(formulation.get("constraints", []), 1):
        lines.append(f"  {i}. {constraint.get('equation')}")
        if constraint.get('description'):
            lines.append(f"     ({constraint.get('description')})")
    
    obj = formulation.get("objective", {})
    lines.append(f"\nOBJECTIVE: {obj.get('type', 'unknown').upper()}")
    lines.append(f"  {obj.get('expression', 'N/A')}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

async def run_complex_workflow(
    problem: str,
    formulator_llm: BaseChatModel,
    equation_checker_llm: BaseChatModel,
    translator_llm: BaseChatModel,
    code_checker_llm: BaseChatModel,
    solver_executor_llm: BaseChatModel,
    validate_tool: Any,
    solve_tool: Any,
    max_iterations: int = 10,
    verbose: bool = False,
) -> dict:
    """
    Run the complex 5-agent workflow.
    
    Args:
        problem: Problem description
        formulator_llm: LLM for mathematical formulation
        equation_checker_llm: LLM for validating equations
        translator_llm: LLM for MiniZinc translation
        code_checker_llm: LLM for validating MiniZinc code
        solver_executor_llm: LLM for diagnosing solver errors
        validate_tool: MiniZinc validation tool
        solve_tool: MiniZinc solve tool
        max_iterations: Max workflow iterations
        verbose: Print detailed output
    
    Returns:
        dict with results
    """
    state = WorkflowState(problem=problem)
    
    results = {
        "problem": problem,
        "iterations": 0,
        "success": False,
        "final_response": None,
        "formulation": None,
        "mzn_code": None,
        "solution": None,
        "workflow_trace": []  # Track which agents were invoked
    }
    
    if verbose:
        print("=" * 80)
        print("COMPLEX 5-AGENT WORKFLOW")
        print("=" * 80)
        print(f"Problem: {problem[:150]}...")
        print()
    
    for iteration in range(max_iterations):
        state.iteration = iteration + 1
        results["iterations"] = iteration + 1
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1} - Phase: {state.phase.upper()}")
            print(f"{'='*80}")
        
        # ========== PHASE 1: FORMULATION ==========
        if state.phase == "formulation":
            if verbose:
                print("\n[1. FORMULATOR AGENT]")
            
            results["workflow_trace"].append(f"Iter{iteration+1}:Formulator")
            
            formulator_messages = [SystemMessage(content=FORMULATOR_SYSTEM_PROMPT)]
            
            if state.last_feedback:
                formulator_messages.append(HumanMessage(
                    content=f"Problem:\n{problem}\n\nFeedback on previous formulation:\n{state.last_feedback.get('message', 'Please revise')}"
                ))
            else:
                formulator_messages.append(HumanMessage(
                    content=f"Create a mathematical formulation for this optimization problem:\n\n{problem}"
                ))
            
            LOGGER.debug(f"=== FORMULATOR INPUT (Iteration {iteration + 1}) ===")
            LOGGER.debug(formulator_messages[-1].content[:500])
            
            formulator_response = await formulator_llm.ainvoke(formulator_messages)
            content_str = stringify_content(formulator_response.content)
            
            LOGGER.debug(f"=== FORMULATOR OUTPUT ===")
            LOGGER.debug(content_str[:500])
            
            formulator_json = extract_json_from_response(content_str)
            
            if not formulator_json or formulator_json.get("action") != "provide_formulation":
                if verbose:
                    print("ERROR: Formulator did not provide valid formulation")
                continue
            
            state.current_formulation = formulator_json
            state.current_equations = format_formulation_for_display(formulator_json)
            state.phase = "equation_check"
            state.last_feedback = None
            
            if verbose:
                print(f"Formulation created:\n{state.current_equations[:300]}...")
            
            continue
        
        # ========== PHASE 2: EQUATION CHECKING ==========
        if state.phase == "equation_check":
            if verbose:
                print("\n[2. EQUATION CHECKER AGENT]")
            
            results["workflow_trace"].append(f"Iter{iteration+1}:EquationChecker")
            
            checker_messages = [
                SystemMessage(content=EQUATION_CHECKER_SYSTEM_PROMPT),
                HumanMessage(content=f"Original problem:\n{problem}\n\nProposed formulation:\n{state.current_equations}\n\nVerify this formulation.")
            ]
            
            LOGGER.debug(f"=== EQUATION CHECKER INPUT ===")
            LOGGER.debug(checker_messages[-1].content[:500])
            
            checker_response = await equation_checker_llm.ainvoke(checker_messages)
            content_str = stringify_content(checker_response.content)
            
            LOGGER.debug(f"=== EQUATION CHECKER OUTPUT ===")
            LOGGER.debug(content_str[:500])
            
            checker_json = extract_json_from_response(content_str)
            
            if not checker_json:
                if verbose:
                    print("ERROR: Equation Checker response parsing failed")
                continue
            
            if checker_json.get("action") == "approve":
                if verbose:
                    print("✓ Equations approved, proceeding to translation")
                state.phase = "translation"
                state.last_feedback = None
                results["formulation"] = state.current_formulation
            else:
                if verbose:
                    print(f"✗ Equations rejected: {checker_json.get('reasoning', 'Unknown')}")
                state.phase = "formulation"
                state.last_feedback = {
                    "message": "\n".join(checker_json.get("issues", ["Please revise"]))
                }
            
            continue
        
        # ========== PHASE 3: TRANSLATION ==========
        if state.phase == "translation":
            if verbose:
                print("\n[3. TRANSLATOR AGENT]")
            
            results["workflow_trace"].append(f"Iter{iteration+1}:Translator")
            
            translator_messages = [SystemMessage(content=TRANSLATOR_SYSTEM_PROMPT)]
            
            if state.last_feedback:
                translator_messages.append(HumanMessage(
                    content=f"Mathematical formulation:\n{state.current_equations}\n\nFeedback on previous MiniZinc code:\n{state.last_feedback.get('message', 'Please fix')}"
                ))
            else:
                translator_messages.append(HumanMessage(
                    content=f"Translate this mathematical formulation into MiniZinc:\n\n{state.current_equations}"
                ))
            
            LOGGER.debug(f"=== TRANSLATOR INPUT ===")
            LOGGER.debug(translator_messages[-1].content[:500])
            
            translator_response = await translator_llm.ainvoke(translator_messages)
            content_str = stringify_content(translator_response.content)
            
            LOGGER.debug(f"=== TRANSLATOR OUTPUT ===")
            LOGGER.debug(content_str[:500])
            
            translator_json = extract_json_from_response(content_str)
            
            if not translator_json or translator_json.get("action") != "provide_code":
                if verbose:
                    print("ERROR: Translator did not provide code")
                continue
            
            state.current_mzn_code = translator_json.get("mzn_code")
            if not state.current_mzn_code:
                if verbose:
                    print("ERROR: MiniZinc code is empty")
                continue
            
            if verbose:
                print(f"MiniZinc code generated:\n{state.current_mzn_code[:200]}...")
            
            # Validate syntax
            validation_result = validate_tool.invoke({"mzn_code": state.current_mzn_code})
            
            if not validation_result.get("valid"):
                if verbose:
                    print(f"✗ Syntax validation failed: {validation_result.get('issues')}")
                state.last_feedback = {
                    "message": "Syntax errors:\n" + "\n".join(validation_result.get("issues", []))
                }
                continue  # Stay in translation phase
            
            if verbose:
                print("✓ Syntax valid")
            
            state.phase = "code_check"
            state.last_feedback = None
            continue
        
        # ========== PHASE 4: CODE CHECKING ==========
        if state.phase == "code_check":
            if verbose:
                print("\n[4. CODE CHECKER AGENT]")
            
            results["workflow_trace"].append(f"Iter{iteration+1}:CodeChecker")
            
            code_checker_messages = [
                SystemMessage(content=CODE_CHECKER_SYSTEM_PROMPT),
                HumanMessage(content=f"Mathematical formulation:\n{state.current_equations}\n\nMiniZinc implementation:\n{state.current_mzn_code}\n\nVerify the code correctly implements the formulation.")
            ]
            
            LOGGER.debug(f"=== CODE CHECKER INPUT ===")
            LOGGER.debug(code_checker_messages[-1].content[:500])
            
            code_checker_response = await code_checker_llm.ainvoke(code_checker_messages)
            content_str = stringify_content(code_checker_response.content)
            
            LOGGER.debug(f"=== CODE CHECKER OUTPUT ===")
            LOGGER.debug(content_str[:500])
            
            code_checker_json = extract_json_from_response(content_str)
            
            if not code_checker_json:
                if verbose:
                    print("ERROR: Code Checker response parsing failed")
                continue
            
            if code_checker_json.get("action") == "approve":
                if verbose:
                    print("✓ Code approved, proceeding to solver")
                state.phase = "solving"
                state.last_feedback = None
                results["mzn_code"] = state.current_mzn_code
            else:
                if verbose:
                    print(f"✗ Code rejected: {code_checker_json.get('reasoning', 'Unknown')}")
                state.phase = "translation"
                state.last_feedback = {
                    "message": "\n".join(code_checker_json.get("issues", ["Please fix"]))
                }
            
            continue
        
        # ========== PHASE 5: SOLVING ==========
        if state.phase == "solving":
            if verbose:
                print("\n[5. SOLVER EXECUTION]")
            
            results["workflow_trace"].append(f"Iter{iteration+1}:Solver")
            
            solve_result = await solve_tool.invoke_async({"mzn_code": state.current_mzn_code})
            
            if verbose:
                print(f"Solver status: {solve_result.get('run_status')}")
            
            if solve_result.get("run_status") == "success":
                results["success"] = True
                results["solution"] = solve_result.get("variables", {})
                results["final_response"] = (
                    f"Optimal solution found:\n{solve_result.get('variables', {})}\n\n"
                    f"Summary: {solve_result.get('summary', 'Success')}"
                )
                
                if verbose:
                    print(f"✓ SUCCESS: Solution found")
                
                return results
            else:
                # Solver failed - diagnose and route error
                if verbose:
                    print(f"✗ Solver failed: {solve_result.get('summary')}")
                    print("\n[SOLVER EXECUTOR - Error Diagnosis]")
                
                results["workflow_trace"].append(f"Iter{iteration+1}:SolverExecutor")
                
                # Use solver executor LLM to diagnose
                executor_messages = [
                    SystemMessage(content=SOLVER_EXECUTOR_PROMPT),
                    HumanMessage(content=f"Solver error occurred:\n{solve_result.get('error', 'Unknown error')}\n\nClassify and provide feedback.")
                ]
                
                executor_response = await solver_executor_llm.ainvoke(executor_messages)
                executor_json = extract_json_from_response(stringify_content(executor_response.content))
                
                if not executor_json:
                    executor_json = {"error_type": "unknown", "feedback": str(solve_result.get('error', 'Unknown'))}
                
                error_type = executor_json.get("error_type", "unknown")
                feedback = executor_json.get("feedback", "Please fix the error")
                
                if verbose:
                    print(f"Diagnosed error type: {error_type}")
                    print(f"Routing feedback...")
                
                if error_type == "minizinc_syntax":
                    # Route to translator
                    state.phase = "translation"
                    state.last_feedback = {"message": f"MiniZinc error: {feedback}"}
                elif error_type == "mathematical":
                    # Route to formulator
                    state.phase = "formulation"
                    state.last_feedback = {"message": f"Mathematical formulation issue: {feedback}"}
                else:
                    # Default: route to translator (most common case)
                    state.phase = "translation"
                    state.last_feedback = {"message": f"Solver error: {feedback}"}
                
                continue
    
    # Max iterations reached
    results["final_response"] = f"Failed to converge after {max_iterations} iterations. Last phase: {state.phase}"
    results["workflow_trace"].append(f"Failed:MaxIterations:{state.phase}")
    return results
