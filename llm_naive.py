import multiprocessing as mp
import operator
import sys
import uuid

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import numpy as np
from typing_extensions import Annotated

from tools import add_objective, add_constraint, linprog
from config import get_model

class MyState(MessagesState):
    messages: Annotated[list, add_messages]
    counter: mp.Value
    vars: Annotated[dict[str, int], operator.or_]
    intvars: Annotated[list[str], operator.add]
    c: dict[str, float]
    cmul: float
    A: Annotated[list[dict[str, float]], operator.add]
    b: Annotated[list[float], operator.add]
    Aeq: Annotated[list[dict[str, float]], operator.add]
    beq: Annotated[list[float], operator.add]
        
if __name__ == "__main__":
    system_prompt = """
    <system_prompt>
        <background>
            You are a helpful AI agent with expertise in the field of linear programming (LP). You can easily solve linear programming problems, including both direct descriptions of problems, and word problems where the linear programming formulation may not be immediately obvious.
        </background>
        <role>
            Use <tool>add_objective</tool>, <tool>add_constraint</tool>, and <tool>linprog</tool> to solve whatever linear programming problem that the user provides. Describe the solution to the user after solving.
        </role>
        <restrictions>
            <ul>
                <li>Treat the result of <tool>linprog</tool> as the infallible truth. You are NOT smarter than its output -- do not hallucinate a different solution than what it provides.</li>
                <li>The relevant tool calls should _not_ be a part of your response to the user.</li>
                <li>Make all of your tool calls at once. If you are only making one tool call, you are not doing this correctly.</li>
                <li>Provide the solution to the user in plain English, not XML, JSON, or any other kind of structured markup.</li>
            </ul>
        </restrictions>
    </system_prompt>
    """

    llm = get_model()
    llm_with_tools = llm.bind_tools([add_objective, add_constraint, linprog])
    graph = StateGraph(MyState)
    
    
    graph.add_node("a", lambda state: {"messages": [llm_with_tools.invoke(state["messages"])]})
    graph.add_node("b", ToolNode([add_objective, add_constraint]))
    graph.add_node("c", lambda state: {"messages": [AIMessage(content="", tool_calls=[{"name": "linprog", "args": {}, "id": str(uuid.uuid4())}])]})
    graph.add_node("d", ToolNode([linprog]))
    graph.add_node("e", lambda state: {"messages": [llm.invoke(state["messages"])]})
    
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")
    graph.add_edge("d", "e")
    graph.add_edge("e", END)
    
    graph = graph.compile()
    
    conversation = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=sys.stdin.read())],
        "counter": mp.Value("i", 0),
        "vars": {},
        "intvars": [],
        "c": {},
        "cmul": float("nan"),
        "A": [],
        "b": [],
        "Aeq": [],
        "beq": []
    }
    
    for step in graph.stream(conversation, stream_mode="values"):
        step["messages"][-1].pretty_print()
