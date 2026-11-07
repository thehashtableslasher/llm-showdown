#!/usr/bin/env python3

import copy
import uuid

import andes
from langchain_core.runnables import RunnableLambda
from langchain.messages import AIMessage
from langchain.tools import tool, ToolRuntime

@tool
def run_tds(runtime: ToolRuntime) -> list[float]:
    '''
    Run a time-domain simulation on the model. Returns the final values of OMEGA.
    '''
    ss = runtime.state.get("ss")
    
    if ss is None:
        pass # TODO handle error
        
    ss = copy.deepcopy(ss)
    ss.setup()
    ss.PFlow.run()
    ss.TDS.run()
    
    return ss.GENROU.omega.v.tolist()
    
class RunTDSNode(RunnableLambda):
    def __init__(self):
        super().__init__(lambda _: {"messages": [AIMessage(content="", tool_calls=[{"name": "run_tds", "args": {}, "id": str(uuid.uuid4())}])]})
