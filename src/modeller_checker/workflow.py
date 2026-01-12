"""
Modeller-Checker workflow core implementation.

This module provides the core workflow logic that can be used by:
- MCP server (stdio/HTTP)
- LangChain tools
- CLI scripts
"""

import sys
from pathlib import Path
import json
import re
import logging
from typing import Optional, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel

# Add src to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

LOGGER = logging.getLogger("modeller_checker.workflow")


MODELLER_SYSTEM_PROMPT = """You are an expert problem analyst and MiniZinc modelling assistant. Your role is to:
1. Take a problem description from the user
2. Create or refine a MiniZinc model that represents the problem
3. Ensure the model has correct MiniZinc syntax (must use 'solve maximize' or 'solve minimize', not just 'maximize' or 'minimize')
4. Include clear variable definitions, constraints, and an objective
5. Use proper MiniZinc syntax: variables, constraints block, then solve statement

CRITICAL: MiniZinc requires:
- Variables declared with 'var type: name;'
- Constraints in a 'constraint ...' block or individual constraint statements
- Objective MUST use 'solve maximize' or 'solve minimize' (not just 'maximize'/'minimize')
- Proper semicolons and syntax

When the Checker provides feedback on your model:
- Fix identified issues carefully
- Do NOT change the core logic unless explicitly requested
- Provide improved MiniZinc code that addresses the feedback

Respond with a JSON object containing:
{
    "action": "provide_model",
    "mzn_code": "...",
    "explanation": "brief explanation of your model"
}

Always provide complete, runnable MiniZinc code that will pass syntax validation."""


CHECKER_SYSTEM_PROMPT = """You are an expert problem analyst and MiniZinc verifier. Your role is to:
1. Review the provided MiniZinc model
2. Verify it correctly models the problem statement
3. Check that constraints and objective match the problem
4. Verify the code uses proper MiniZinc syntax (solve maximize/minimize, proper declarations, etc.)
5. Either approve the model or provide specific feedback for improvement

Respond with a JSON object containing:
{
    "action": "approve" | "reject",
    "reasoning": "explanation of your analysis",
    "issues": ["issue1", "issue2", ...] (only if action is "reject")
}

Be precise: if the model has issues (semantic or syntactic), list exactly what needs to be fixed."""


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
            # Clean up newlines and concatenation artifacts
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


async def run_modeller_checker_workflow(
    problem: str,
    modeller_llm: BaseChatModel,
    checker_llm: BaseChatModel,
    validate_tool: Any,
    solve_tool: Any,
    max_iterations: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Run the modeller-checker workflow.
    
    Args:
        problem: Problem description to model
        modeller_llm: LLM instance for the modeller agent
        checker_llm: LLM instance for the checker agent
        validate_tool: MiniZinc validation tool
        solve_tool: MiniZinc solve tool
        max_iterations: Max modeller-checker rounds before giving up
        verbose: Print detailed output
    
    Returns:
        dict with final_response, mzn_code, iterations, success, checker_approval
    """
    results = {
        "problem": problem,
        "iterations": 0,
        "success": False,
        "final_response": None,
        "mzn_code": None,
        "checker_approval": False,
    }
    
    if verbose:
        print("=" * 80)
        print("MODELLER-CHECKER WORKFLOW")
        print("=" * 80)
        print(f"Problem: {problem[:100]}...")
        print()
    
    # Track MiniZinc code across iterations
    current_mzn_code = None
    current_feedback = None
    
    for iteration in range(max_iterations):
        results["iterations"] = iteration + 1
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*80}")
        
        # ========== MODELLER PHASE ==========
        if verbose:
            print("\n[MODELLER PHASE]")
        
        modeller_messages: list[BaseMessage] = [
            SystemMessage(content=MODELLER_SYSTEM_PROMPT),
        ]
        
        if iteration == 0:
            modeller_messages.append(HumanMessage(content=f"Problem to model:\n{problem}"))
        else:
            # Format feedback issues (can be strings or dicts with 'message' field)
            feedback_lines = []
            for issue in current_feedback["issues"]:
                if isinstance(issue, dict):
                    feedback_lines.append(issue.get("message", str(issue)))
                else:
                    feedback_lines.append(str(issue))
            feedback_text = "\n".join(feedback_lines)
            
            modeller_messages.append(
                HumanMessage(
                    content=(
                        f"The Checker found issues with the MiniZinc model:\n{feedback_text}\n\n"
                        "Please fix these issues and provide an improved model."
                    )
                )
            )
        
        if verbose:
            print(f"Modeller input: {modeller_messages[-1].content[:200]}...")
        
        LOGGER.debug(f"=== MODELLER INPUT (Iteration {iteration + 1}) ===")
        LOGGER.debug(f"System: {MODELLER_SYSTEM_PROMPT[:200]}...")
        LOGGER.debug(f"User: {modeller_messages[-1].content}")
        
        modeller_response: AIMessage = await modeller_llm.ainvoke(modeller_messages)
        
        content_str = stringify_content(modeller_response.content)
        LOGGER.debug(f"=== MODELLER OUTPUT (Iteration {iteration + 1}) ===")
        LOGGER.debug(f"{content_str}")
        
        if verbose:
            print(f"Modeller response:\n{content_str[:500]}\n")
        
        # Parse modeller response
        modeller_json = extract_json_from_response(
            stringify_content(modeller_response.content)
        )
        
        if not modeller_json or modeller_json.get("action") != "provide_model":
            if verbose:
                print("ERROR: Modeller did not provide a model. Retrying...")
            continue
        
        current_mzn_code = modeller_json.get("mzn_code")
        if not current_mzn_code:
            if verbose:
                print("ERROR: Modeller mzn_code is empty. Retrying...")
            continue
        
        # Validate syntax
        if verbose:
            print("\n[SYNTAX VALIDATION]")
        
        validation_result = validate_tool.invoke({"mzn_code": current_mzn_code})
        
        if verbose:
            print(f"Validation: {validation_result}")
        
        if not validation_result.get("valid"):
            if verbose:
                issues = validation_result.get("issues", [])
                print(f"Syntax errors found: {issues}")
                print("Sending feedback to modeller...")
            
            # Create feedback for modeller about syntax
            current_feedback = {
                "issues": validation_result.get("issues", ["Syntax error, please fix"])
            }
            continue
        
        if verbose:
            print("✓ Syntax is valid")
        
        # ========== CHECKER PHASE ==========
        if verbose:
            print("\n[CHECKER PHASE]")
        
        # Include validation info in checker context
        validation_info = ""
        if validation_result.get("issues"):
            validation_info = f"\n\nNOTE: Syntax validation found issues:\n" + "\n".join(
                f"  - {issue}" for issue in validation_result.get("issues", [])
            )
        
        checker_messages: list[BaseMessage] = [
            SystemMessage(content=CHECKER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Original problem:\n{problem}\n\n"
                    f"Proposed MiniZinc model:\n{current_mzn_code}{validation_info}\n\n"
                    f"Please verify if this model correctly represents the problem and is syntactically correct."
                )
            ),
        ]
        
        if verbose:
            print(f"Checker analyzing model against problem...")
        
        LOGGER.debug(f"=== CHECKER INPUT (Iteration {iteration + 1}) ===")
        LOGGER.debug(f"System: {CHECKER_SYSTEM_PROMPT[:200]}...")
        LOGGER.debug(f"User: {checker_messages[-1].content}")
        
        checker_response: AIMessage = await checker_llm.ainvoke(checker_messages)
        
        content_str = stringify_content(checker_response.content)
        LOGGER.debug(f"=== CHECKER OUTPUT (Iteration {iteration + 1}) ===")
        LOGGER.debug(f"{content_str}")
        
        if verbose:
            print(f"Checker response:\n{content_str[:500]}\n")
        
        # Parse checker response
        checker_json = extract_json_from_response(
            stringify_content(checker_response.content)
        )
        
        if not checker_json:
            if verbose:
                print("ERROR: Checker response parsing failed. Retrying...")
            continue
        
        action = checker_json.get("action", "").lower()
        
        if action == "approve":
            if verbose:
                print("✓ CHECKER APPROVED - Proceeding to solve")
            
            results["checker_approval"] = True
            results["mzn_code"] = current_mzn_code
            
            # ========== SOLVE PHASE ==========
            if verbose:
                print("\n[SOLVE PHASE]")
            
            solve_result = await solve_tool.invoke_async({"mzn_code": current_mzn_code})
            
            if verbose:
                print(f"Solve result: {solve_result}")
            
            if solve_result.get("run_status") == "success":
                results["success"] = True
                variables = solve_result.get("variables", {})
                results["final_response"] = (
                    f"Solver found optimal solution:\n{variables}\n\n"
                    f"Summary: {solve_result.get('summary', 'Solution found')}"
                )
                
                if verbose:
                    print(f"\n✓ SUCCESS: {results['final_response']}")
                
                return results
            else:
                # Solver failed - feed error back to checker/modeller for another iteration
                if verbose:
                    print(f"✗ Solver failed - {solve_result.get('summary', 'Unknown error')}")
                    print("Feeding solver error back to modeller for correction...")
                
                # Extract error details
                error_summary = solve_result.get('summary', 'Unknown solver error')
                error_details = solve_result.get('error', error_summary)
                
                # Create feedback that will be sent to modeller
                current_feedback = {
                    "issues": [
                        f"The MiniZinc solver reported an error when executing the model: {error_details}",
                        "Please fix the model to resolve this solver error."
                    ]
                }
                
                # Continue to next iteration instead of returning
                continue
        
        elif action == "reject":
            issues = checker_json.get("issues", ["Unknown issue"])
            if verbose:
                print(f"✗ CHECKER REJECTED")
                for issue in issues:
                    print(f"  - {issue}")
            
            current_feedback = {"issues": issues}
            # Loop continues to next iteration with feedback
        
        else:
            if verbose:
                print(f"ERROR: Unknown checker action '{action}'. Retrying...")
            continue
    
    results["final_response"] = f"Failed to converge after {max_iterations} iterations."
    return results
