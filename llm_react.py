import operator
import sys
import uuid

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from tools import add_objective, add_constraint, linprog
from config import get_model

concat = lambda a, b: np.concat((a, b)) if a is not None and b is not None else b

class MyState(MessagesState):
    messages: Annotated[list, add_messages]
    c: np.ndarray
    cmul: float
    A: Annotated[np.ndarray, concat]
    b: Annotated[np.ndarray, concat]
    Aeq: Annotated[np.ndarray, concat]
    beq: Annotated[np.ndarray, concat]
    next_step: str
        
background = """
You are a helpful AI agent with expertise in the field of linear programming 
(LP). You can easily solve linear programming problems, including both direct 
descriptions of problems, and word problems where the linear programming 
formulation may not be immediately obvious.
"""

llm = get_model()

def thinker(state):
    class ThinkerResult(BaseModel):
        next_step: Literal["add_objective", "add_constraint", "linprog", "done"] = Field(..., description="The next action to take.")
        justification: str = Field(..., description="A brief (< 100 words) rationale of what led you to choose the next step.")

    system_prompt = """
    <system_prompt>
        <background>{background}</background>
        <role>
            Determine the next step to solving the linear programming problem, based on the steps that have already been performed.
            <step name="add_objective">Set the objective function of the linear programming problem.</step>
            <step name="add_constraint">Add a constraint to the linear programming problem.</step>
            <step name="linprog">Solve the linear programming problem that has been constructed by the added objective and constraints</step>
            <step name="done">Wrap everything up and report the results to the user</step>
        </role>
        <restrictions>
            <ul>
                <li>Do not provide nothing as a response -- your output MUST be a valid JSON object that conforms to the provided schema.</li>
                <li>You should always try to add a constraint or objective BEFORE solving the linear programming problem.</li>
            </ul>
        </restrictions>
    </system_prompt>
    """.format(background=background)
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    #llm_structured = llm.with_structured_output(ThinkerResult, include_raw=True)
    #message = llm_structured.invoke(messages)
    
    while True:
        llm_structured = llm.with_structured_output(ThinkerResult, include_raw=True)
        message = llm_structured.invoke(messages)
        if message["parsed"] is not None:
            break
            
        print("GOT AN INVALID RESPONSE", message["raw"])
        messages.extend([
            message["raw"],
            HumanMessage(content="That was an invalid response. Please follow your system prompt more closely and try again.")
        ])
    
    return {"messages": [message["raw"]], "next_step": message["parsed"].next_step}

def add_objective_node(state):
    system_prompt = """
    <system_prompt>
        <background>{background}</background>
        <role>
            Call <tool>add_objective</tool> with the objective function for the specified linear programming problem.
        </role>
        <restrictions>
            <ul>
                <li>Do NOT tell the user anything -- that is NOT your job. Your job is to call <tool>add_objective</tool>.</li>
            </ul>
        </restrictions>
    </system_prompt>
    """.format(background=background)
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    #llm_with_tools = llm.bind_tools([add_objective])
    #message = llm_with_tools.invoke(messages)
    
    while True:
        llm_with_tools = llm.bind_tools([add_objective])
        message = llm_with_tools.invoke(messages)
        if message.tool_calls:
            break
            
        print("GOT AN INVALID RESPONSE", message)
        messages.extend([
            message,
            HumanMessage(content="That was an invalid response. Please follow your system prompt more closely and try again.")
        ])
    
    return {"messages": [message]}

def add_constraint_node(state):
    system_prompt = """
    <system_prompt>
        <background>{background}</background>
        <role>
            Call <tool>add_constraint</tool> for one or more of the constraints that have not yet been added by a previous call to <tool>add_constraint</tool>.
        </role>
        <restrictions>
            <ul>
                <li>Do NOT tell the user anything -- that is NOT your job. Your job is to call <tool>add_constraint</tool>.</li>
                <li>"less_than_or_equal_to" is NOT A VALID CONSTRAINT PARAMETER. Use "less_than".</li>
            </ul>
        </restrictions>
    </system_prompt>
    """.format(background=background)
    
    #llm_with_tools = llm.bind_tools([add_objective])
    #message = llm_with_tools.invoke(messages)
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    while True:
        llm_with_tools = llm.bind_tools([add_constraint])
        message = llm_with_tools.invoke(messages)
        if message.tool_calls:
            break
            
        print("GOT AN INVALID RESPONSE", message)
        messages.extend([
            message,
            HumanMessage(content="That was an invalid response. Please follow your system prompt more closely and try again.")
        ])
    
    return {"messages": [message]}

def linprog_node(state):
    return {"messages": [AIMessage(content="", tool_calls=[{"name": "linprog", "args": {}, "id": str(uuid.uuid4())}])]}

def done(state):
    system_prompt = """
    <system_prompt>
        <background>{background}</background>
        <role>
            Describe the solution to the linear programming problem to the user. Use previous calls to <tool>linprog</tool> as context.
        </role>
        <restrictions>
            <ul>
                <li>Treat the result of <tool>linprog</tool> as the infallible truth. You are NOT smarter than its output -- do not hallucinate a different solution than what it provides. This includes treating the justification of the previous message as the valid result, when it does not match the result of <tool>linprog</tool>.</li>
            </ul>
        </restrictions>
    </system_prompt>
    """.format(background=background)
    
    #message = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    while True:
        llm_with_tools = llm.bind_tools([add_objective])
        message = llm_with_tools.invoke(messages)
        if message.content:
            break
            
        print("GOT AN INVALID RESPONSE", message)
        messages.extend([
            message,
            HumanMessage(content="That was an invalid response. Please follow your system prompt more closely and try again.")
        ])
    
    return {"messages": [message]}

if __name__ == "__main__":

    llm_with_tools = llm.bind_tools([add_objective, add_constraint, linprog])
    graph = StateGraph(MyState)
    
    graph.add_node("thinker", thinker)
    graph.add_node("add_objective", add_objective_node)
    graph.add_node("add_constraint", add_constraint_node)
    graph.add_node("linprog", linprog_node)
    graph.add_node("tools", ToolNode([add_objective, add_constraint, linprog]))
    graph.add_node("done", done)
    
    graph.add_edge(START, "thinker")
    graph.add_conditional_edges("thinker", operator.itemgetter("next_step"))
    graph.add_edge("add_objective", "tools")
    graph.add_edge("add_constraint", "tools")
    graph.add_edge("linprog", "tools")
    graph.add_edge("tools", "thinker")
    graph.add_edge("done", END)
    
    graph = graph.compile()
    
    conversation = {"messages": [HumanMessage(content=sys.stdin.read())]}
    
    for step in graph.stream(conversation, stream_mode="values"):
        step["messages"][-1].pretty_print()
