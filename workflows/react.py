#!/usr/bin/env python3

import operator
import sys
import multiprocessing as mp
import uuid

import andes
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from tools import *

class ReActState(MessagesState):
    messages: Annotated[list, add_messages]
    ss: str
    next_step: str

class ThinkerResult(BaseModel):
    next_step: Literal["add_fault", "add_trip", "load_model", "run_tds", "done"] = Field(..., description="The next action to take.")
    justification: str = Field(..., description="A brief (< 100 words) rationale of what led you to choose the next step.")

def react_graph(llm):
    background = BACKGROUND = """
    You are an AI assistant for ANDES, a Python toolkit for power system modeling 
    and simulation. You are to simultaneously act as an expert on power systems 
    engineering, as well as an interface for manipulating an ANDES simulation. The 
    user will provide instructions or commands related to manipulating or querying 
    some ANDES system. You will be provided additional instructions and tools that 
    will assist you in fulfilling that role as these user instructions are made. 
    Under any of those circumstances, you are to follow the guidelines and 
    constraints laid out here.
    """

    def thinker(state):
        system_prompt = """
        <system_prompt>
            <background>{background}</background>

            <role>
                Determine the next step to fufill the user's ANDES request, based on the steps that have already been performed.
                
                <category name="load_model">Load a local model or test case.</category>
                <category name="add_fault">Add a bus fault to the loaded model.</category>
                <category name="add_trip">Add a device trip to the loaded model.</category>
                <category name="run_tds">Run a time-domain simulation on an ALREADY LOADED model.</category>
                <category name="done">Wrap everything up and report the results to the user.</category>
            </role>
            
            <constraints>
                <constraint>Do not provide any additional output, aside from the next step and justification.</constraint>
                <constraint>Do not provide nothing as a response -- your output MUST be a valid JSON object that conforms to the provided schema.</constraint>
                <constraint>A model MUST be loaded before using the "run_tds" step. Do not attempt to go to that step if a "load_model" step isn't present in the conversation.</constraint>
            </constraints>
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

    def done(state):
        system_prompt = """
        <system_prompt>
            <background>{background}</background>
            <role>
                Describe the solution to the original prompt to the user. Use previous tool calls and decision-making as context.
            </role>
            <restrictions>
                <ul>
                    <li>Treat the result of tool calls as the infallible truth. You are NOT smarter than its output -- do not hallucinate a different solution than what it provides. This includes treating the justification of the previous message as the valid result, when it does not match the result of your tool calls.</li>
                </ul>
            </restrictions>
        </system_prompt>
        """.format(background=background)
        
        message = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        
        return {"messages": [message]}
        
    graph = StateGraph(ReActState)
    
    graph.add_node("thinker", thinker)
    graph.add_node("load_model", LoadModelNode(llm))
    graph.add_node("add_fault", AddFaultNode(llm))
    graph.add_node("add_trip", AddTripNode(llm))
    graph.add_node("run_tds", RunTDSNode())
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("done", done)
    
    graph.add_edge(START, "thinker")
    graph.add_conditional_edges("thinker", operator.itemgetter("next_step"))
    graph.add_edge("load_model", "tools")
    graph.add_edge("add_fault", "tools")
    graph.add_edge("add_trip", "tools")
    graph.add_edge("run_tds", "tools")
    graph.add_edge("tools", "thinker")
    graph.add_edge("done", END)
    
    graph = graph.compile()
    
    conversation = {
        "messages": [],
        "ss": None,
        "next_step": ""
    }
    
    return graph, conversation
