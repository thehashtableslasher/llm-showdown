from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
import numpy as np
from scipy.optimize import linprog as splinprog
from typing_extensions import Literal, Optional, TypedDict

@tool
def add_decision_variable(name: str, is_integer: bool = False, runtime: ToolRuntime) -> Command:
    """
    <tool>
        Add a new decision variable to the linear programming problem. Decision variables need to be added before they are referenced by <tool>add_objective</tool> or <tool>add_constraint</tool>.
        <parameter name="name">A unique identifying name for this decision variable.</parameter>
        <parameter name="is_integer">Whether or not this decision variable should be restricted to an integer.</parameter>
    </tool>
    """
    
    state = runtime.state
    update = {}
    
    if name in state["vars"]:
        return Command(update={"messages": [ToolMessage(content=f"Failure: decision variable \"{name}\" already exists", tool_call_id=runtime.tool_call_id)]})
    
    with state["counter"].get_lock():
        update["vars"] = {name: state["counter"].value}
        state["counter"].value += 1
        
    if is_integer:
        update["intvars"] = [name]
        
    update["messages"] = [ToolMessage(content="Success", tool_call_id=runtime.tool_call_id)]
    
    return Command(update=update)

@tool
def add_objective(coefficients: dict[str, float], kind: Literal["minimize", "maximize"], runtime: ToolRuntime) -> Command:
    """
    <tool>
        Set the objective function for the current linear programming problem.
        <parameter name="coefficients">Mapping of variables to their scalar coefficients in the objective function. Variables must be created by <tool>add_decision_variable</tool> before use. Any variables absent from this mapping will automatically be zeroed.</parameter>
        <parameter name="kind">Whether or not we seek to minimize or maximize the objective function.</parameter>
    </tool>
    """
    
    state = runtime.state
    update = {}
    
    for name in coefficients.keys():
        if name not in state["vars"]:
            return Command(update={"messages": [ToolMessage(content=f"Failure: decision variable \"{name}\" does not exist", tool_call_id=runtime.tool_call_id)]})
        
    if kind == "minimize":
        update["c"] = coefficients
        update["cmul"] = 1.0
    elif kind == "maximize":
        update["c"] = {name: -coeff for name, coeff in coefficients}
        update["cmul"] = -1.0
    else:
        return Command(update={"messages": [ToolMessage(content="Failure: kind is not one of {\"minimize\", \"maximize\"}", tool_call_id=runtime.tool_call_id)]})
        
    update["messages"] = [ToolMessage(content="Success", tool_call_id=runtime.tool_call_id)]
    
    return Command(update=update)

@tool
def add_constraint(coefficients: dict[str, float], value: float, kind: Literal["less_than_or_equal", "equal", "greater_than_or_equal"], runtime: ToolRuntime) -> Command:
    """
    <tool>
        Add a new constraint to the current linear programming problem.
        <parameter name="coefficients">Mapping of variables to their scalar coefficients in this specific constraint. Variables must be created by <tool>add_decision_variable</tool> before use.</parameter>
        <parameter name="value">The numeric limitation placed on the linear combination of the variables for this constraint</parameter>
        <parameter name="kind">If this contraint is to place an upper bound, (a less-than-or-equal-to relation), a lower bound (a greater-than-or-equal-to relation), or an exact value (an equal-to relation) on this constraint, relative to the value.</parameter>
    </tool>
    """
    
    state = runtime.state
    update = {}
    
    for name in coefficients.keys():
        if name not in state["vars"]:
            return Command(update={"messages": [ToolMessage(content=f"Failure: decision variable \"{name}\" does not exist", tool_call_id=runtime.tool_call_id)]})
    
    if kind == "less_than":
        update["A"] = [coefficients]
        update["b"] = [value]
    elif kind == "equal":
        update["Aeq"] = [coefficients]
        update["beq"] = [value]
    elif kind == "greater_than":
        update["A"] = [{name: -coeff for name, coeff in coefficients}]
        update["b"] = [-value]
    else:
        return Command(update={"messages": [ToolMessage(content="Failure: kind is not one of {\"less_than_or_equal\", \"equal\", \"greater_than_or_equal\"}", tool_call_id=runtime.tool_call_id)]})
    
    update["messages"] = [ToolMessage(content="Success", tool_call_id=runtime.tool_call_id)]
    
    return Command(update=update)

class LinprogResult(TypedDict):
    status: Literal["feasible", "infeasible", "unbounded", "unknown_error"]
    objective: Optional[dict[str, float]]
    objective_value: Optional[float]

@tool(response_format="content_and_artifact")
def linprog(runtime: ToolRuntime) -> LinprogResult:
    """
    <tool>
        Compute the solution of the linear programming problem that was earlier specified by <tool>add_objective</tool> and <tool>add_constraint</tool>.
        <return name="status">Whether or not the linear programming problem is feasible, or if there is some issue with its formulation (it may be unbounded, infeasible, or the LP solver may have encountered an unknown error</values>
        <return name="objective">If the problem is feasible, then the optimal minimum/maximum for each variable in the problem.</return>
        <return name="objective_value">If the problem is feasible, then the value of the objective function at the optimal minimum/maximum.</return>
    </tool>
    """
    
    state = runtime.state
    
    var_ct = state["counter"].value
    c = np.zeros(var_ct)
    A = np.zeros((len(state["A"]), var_ct))
    b = np.array(state["b"])
    Aeq = np.zeros((len(state["Aeq"]), var_ct))
    beq = np.array(state["beq"])
    integrality = np.zeros(var_ct)
    
    for name, x in state["c"].items():
        c[state["vars"][name]] = x
    
    for i, row in enumerate(state["A"]):
        for name, x in row.items():
            j = state["vars"][name]
            A[i, j] = x
    
    for i, row in enumerate(state["Aeq"]):
        for name, x in row.items():
            j = state["vars"][name]
            Aeq[i, j] = x
    
    for name in state["intvars"]:
        c[state["vars"][name]] = 1
    
    lpres = splinprog(c, A, b, Aeq, beq, integrality=integrality)
    ret = {}
    
    if lpres.status == 0:
        ret["status"] = "feasible"
        ret["objective"] = {name: lpres.x[i] for name, i in state["vars"].items()}
        ret["objective_value"] = np.dot(c, lpres.x) * state["cmul"]
    elif lpres.status == 2:
        ret["status"] = "infeasible"
    elif lpres.status == 3:
        ret["status"] = "unbounded"
    else:
        ret["status"] = "unknown_error"
    
    return ret, lpres
