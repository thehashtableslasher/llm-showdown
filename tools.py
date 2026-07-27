from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
import numpy as np
from scipy.optimize import linprog as splinprog
from typing_extensions import List, Literal, Optional, TypedDict

@tool
def add_objective(coefficients: List[float], kind: Literal["minimize", "maximize"], runtime: ToolRuntime) -> Command:
    """
    <tool>
        Set the objective function for the current linear programming problem.
        <parameter name="coefficients">Coefficients of each variable in the objective function.</parameter>
        <parameter name="kind">Whether or not we seek to minimize or maximize the objective function.</parameter>
    </tool>
    """
    
    state = runtime.state
    update = {}
    
    if "A" in state and len(coefficients) != state["A"].shape[1]:
        return Command(update={"messages": [ToolMessage(content="Failure", tool_call_id=runtime.tool_call_id)]})
    
    if "Aeq" in state and len(coefficients) != state["Aeq"].shape[1]:
        return Command(update={"messages": [ToolMessage(content="Failure", tool_call_id=runtime.tool_call_id)]})
        
    if kind == "minimize":
        update["c"] = np.array(coefficients)
        update["cmul"] = 1.0
    elif kind == "maximize":
        update["c"] = -np.array(coefficients)
        update["cmul"] = -1.0
    else:
        return Command(update={"messages": [ToolMessage(content="Failure", tool_call_id=runtime.tool_call_id)]})
        
    update["messages"] = [ToolMessage(content="Success", tool_call_id=runtime.tool_call_id)]
    
    return Command(update=update)
    
@tool
def add_constraint(runtime: ToolRuntime) -> Command:
    """
    <tool>
        Clears any constraints or objective that was set previously. Use this before setting up a second linear programming problem, if necessary.
    </tool>
    """
    
    return Command(update={"c": None, "A": None, "b": None, "Aeq": None, "beq": None})

@tool
def add_constraint(coefficients: List[float], value: float, kind: Literal["less_than", "equal", "greater_than"], runtime: ToolRuntime) -> Command:
    """
    <tool>
        Add a new constraint to the current linear programming problem.
        <parameter name="coefficients">Coefficients of each variable in this specific constraint.</parameter>
        <parameter name="value">The numeric limitation placed on the linear combination of the variables for this constraint</parameter>
        <parameter name="kind">If this contraint is to place an upper bound, (a less-than-or-equal-to relation), a lower bound (a greater-than-or-equal-to relation), or an exact value (an equal-to relation) on this constraint, relative to the value. Note that "less_than" includes the less-than-or-equal-to relation -- you should NOT try to force this parameter to be "less_than_or_equal_to".</parameter>
    </tool>
    """
    
    state = runtime.state
    update = {}
    
    if "c" in state and len(coefficients) != state["c"].shape[0]:
        return Command(update={"messages": [ToolMessage(content="Failure", tool_call_id=runtime.tool_call_id)]})
    
    if "A" in state and len(coefficients) != state["A"].shape[1]:
        return Command(update={"messages": [ToolMessage(content="Failure", tool_call_id=runtime.tool_call_id)]})
    
    if "Aeq" in state and len(coefficients) != state["Aeq"].shape[1]:
        return Command(update={"messages": [ToolMessage(content="Failure", tool_call_id=runtime.tool_call_id)]})
        
    row = np.array([coefficients])
    x = np.array([value])
    
    if kind == "less_than":
        update["A"] = row
        update["b"] = x
    elif kind == "equal":
        update["Aeq"] = row
        update["beq"] = x
    elif kind == "greater_than":
        update["A"] = -row
        update["b"] = -x
    else:
        return Command(update={"messages": [ToolMessage(content="Failure", tool_call_id=runtime.tool_call_id)]})
    
    update["messages"] = [ToolMessage(content="Success", tool_call_id=runtime.tool_call_id)]
    
    return Command(update=update)

class LinprogResult(TypedDict):
    status: Literal["feasible", "infeasible", "unbounded", "unknown_error"]
    objective: Optional[List[float]]
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
    
    lpres = splinprog(state["c"], state.get("A"), state.get("b"), state.get("Aeq"), state.get("beq"))
    ret = {}
    
    if lpres.status == 0:
        ret["status"] = "feasible"
        ret["objective"] = lpres.x.tolist()
        ret["objective_value"] = np.dot(state["c"], lpres.x) * state["cmul"]
    elif lpres.status == 2:
        ret["status"] = "infeasible"
    elif lpres.status == 3:
        ret["status"] = "unbounded"
    else:
        ret["status"] = "unknown_error"
    
    return ret, lpres
