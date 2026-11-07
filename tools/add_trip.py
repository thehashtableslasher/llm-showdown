#!/usr/bin/env python3

from langchain.tools import tool, ToolRuntime
from typing_extensions import Literal

from ._common import ReturnStatus, ToolCallerNode

@tool
def add_trip(kind: Literal["GENROU", "PQ", "Line"], device: str, ttrip: float, runtime: ToolRuntime) -> ReturnStatus:
    '''
    Add a device trip to the loaded model.
    
    Parameters:
    * kind: The specific kind of trip to add to the model. Valid options are:
      - GENROU: used to add a generator trip
      - PQ: used to add load shedding
      - Line: used to add a transformer trip
    * device: The device ID to add the trip to.
    * ttrip: The time (seconds) at which the selected device is tripped.
    
    Returns: a status object describing if the call was successful or failed.
    '''

    ss = runtime.state.get("ss")
    
    if ss is None:
        return ReturnStatus.failure("No model is loaded")
    
    if bus not in getattr(ss, kind).idx.v:
        return ReturnStatus.failure("Device ID \"{}\" is not present in the model (valid device IDs for {} are \"{}\")", device, kind, "\", \"".join(getattr(ss, kind).idx.v))
        
    ss.add("Toggle", model=kind, dev=device, t=ttrip)
    
    return ReturnStatus.success()

class AddTripNode(ToolCallerNode):
    TOOLS = [add_trip]
    SYSTEM_MESSAGE = '''
    Use the `add_trip' tool to add a device trip in the given device and at the given time, according to the most recent messages in the conversation history.
    '''
