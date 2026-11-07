#!/usr/bin/env python3

from langchain.tools import tool, ToolRuntime

from ._common import ReturnStatus, ToolCallerNode

@tool
def add_fault(bus: int, tfault: float, tclear: float, runtime: ToolRuntime) -> ReturnStatus:
    '''
    Add a bus fault to the loaded model.
    
    Parameters:
    * bus: The ID of the bus to trigger a fault on.
    * tfault: The time (seconds) at which the fault is triggered.
    * tclear: The time (seconds) at which the fault is resolved.
    
    Returns: a status object describing if the call was successful or failed.
    '''

    ss = runtime.state.get("ss")
    
    if ss is None:
        return ReturnStatus.failure("No model is loaded")
    
    if bus not in ss.Bus.idx.v:
        return ReturnStatus.failure("Bus ID \"{}\" is not present in the model (valid bus IDs are \"{}\")", bus, "\", \"".join(ss.Bus.idx.v))
        
    ss.add("Fault", bus=bus, tf=tfault, tc=tclear)
    
    return ReturnStatus.success()

class AddFaultNode(ToolCallerNode):
    TOOLS = [add_fault]
    SYSTEM_MESSAGE = """
    Use the `add_fault' tool to add a fault in the given bus and at the given times, according to the most recent messages in the conversation history.
    """
