#!/usr/bin/env python3

import json

import andes
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

from ._common import ReturnStatus, ToolCallerNode

@tool
def load_model(filename: str, test_case: bool, runtime: ToolRuntime) -> Command:
    '''
    Load a model in a specified path.
    
    Parameters:
    * filename: The file containing the saved dataframes for the model.
    * test_case: True if the model is a premade test case (this should be considered the default behavior), false otherwise.
    '''
    
    update = {}
    
    try:
        if test_case:
            ss = andes.load(andes.get_case(filename))
        else:
            ss = andes.load(filename)
    except Exception as err:
        update["messages"] = [ToolMessage(
            status="error",
            content=err,
            tool_call_id=runtime.tool_call_id
        )]
    else:   
        update["ss"] = ss
        update["messages"] = [ToolMessage(
            status="success",
            content=ss,
            tool_call_id=runtime.tool_call_id
        )]
        
    return Command(update=update)
    
class LoadModelNode(ToolCallerNode):
    TOOLS = [load_model]
    SYSTEM_MESSAGE = """
    Use the `load_model' tool to load an ANDES model, according to the most recent messages in the conversation history.
    """
