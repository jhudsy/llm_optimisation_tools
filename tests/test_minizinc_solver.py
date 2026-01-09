"""Tests for MiniZinc solver variable extraction and result handling."""

from __future__ import annotations

import pytest

from mzn.solver import MiniZincSolver


def test_minizinc_solver_extracts_variables_and_objective():
    """Test that the solver correctly extracts variable values and objective from result.solution."""
    
    # MiniZinc model for production optimization problem
    mzn_code = """
var float: super;
var float: deluxe;

constraint 0.5*super + 0.25*deluxe <= 120;
constraint 0.5*super + 0.75*deluxe <= 160;
constraint super >= 0;
constraint deluxe >= 0;

solve maximize 23*super + 30*deluxe;
"""
    
    solver = MiniZincSolver(solver_backend="coinbc", time_limit=10.0)
    result = solver.solve(mzn_code)
    
    # Check that the solve was successful
    assert result["run_status"] == "success", f"Expected success, got {result['run_status']}"
    
    # Check that variables were extracted
    assert "variables" in result
    variables = result["variables"]
    assert "super" in variables, f"Variable 'super' not found in {variables}"
    assert "deluxe" in variables, f"Variable 'deluxe' not found in {variables}"
    
    # Check that variables have reasonable values (not checking exact values due to solver variations)
    assert isinstance(variables["super"], (int, float))
    assert isinstance(variables["deluxe"], (int, float))
    assert variables["super"] >= 0
    assert variables["deluxe"] >= 0
    
    # Check that objective value was extracted
    assert result["objective_value"] is not None
    assert isinstance(result["objective_value"], (int, float))
    assert result["objective_value"] > 0
    
    # Check that summary is present
    assert "summary" in result
    assert "OPTIMAL_SOLUTION" in result["summary"]
    

def test_minizinc_solver_simple_problem():
    """Test with a simple two-variable maximization problem."""
    
    mzn_code = """
var 0..10: x;
var 0..10: y;

constraint x + y <= 10;

solve maximize x + 2*y;
"""
    
    solver = MiniZincSolver(solver_backend="coinbc", time_limit=5.0)
    result = solver.solve(mzn_code)
    
    assert result["run_status"] == "success"
    assert "x" in result["variables"]
    assert "y" in result["variables"]
    assert result["objective_value"] is not None
    # Optimal solution should be x=0, y=10 for objective=20
    assert result["objective_value"] == pytest.approx(20.0, rel=0.01)


def test_minizinc_solver_handles_infeasible():
    """Test that solver properly handles infeasible problems."""
    
    mzn_code = """
var 0..5: x;

constraint x >= 10;  % Impossible constraint

solve satisfy;
"""
    
    solver = MiniZincSolver(solver_backend="coinbc", time_limit=5.0)
    
    # Infeasible problems should raise an exception or return failed status
    # The exact behavior depends on the solver backend
    try:
        result = solver.solve(mzn_code)
        # If it doesn't raise, check the status
        assert result["run_status"] != "success"
    except RuntimeError as e:
        # Expected for infeasible problems
        assert "failed" in str(e).lower() or "infeasible" in str(e).lower()


def test_minizinc_solver_multiple_variables():
    """Test extraction of multiple variables from solution."""
    
    mzn_code = """
var float: a;
var float: b;
var float: c;

constraint a + b + c <= 100;
constraint a >= 10;
constraint b >= 20;
constraint c >= 30;

solve maximize a + b + c;
"""
    
    solver = MiniZincSolver(solver_backend="coinbc", time_limit=5.0)
    result = solver.solve(mzn_code)
    
    assert result["run_status"] == "success"
    variables = result["variables"]
    
    # All three variables should be extracted
    assert "a" in variables
    assert "b" in variables
    assert "c" in variables
    
    # Check constraints are satisfied
    assert variables["a"] >= 10
    assert variables["b"] >= 20
    assert variables["c"] >= 30
    assert variables["a"] + variables["b"] + variables["c"] <= 100.01  # Allow small numerical error
