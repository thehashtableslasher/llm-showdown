#!/usr/bin/env python3

from langchain_core.runnables import Runnable
from langchain.messages import HumanMessage, SystemMessage
from typing_extensions import Literal, Optional, TypedDict

class ReturnStatus(TypedDict):
    status: Literal["Success", "Failure"]
    extra: Optional[str]
    
    @staticmethod
    def success():
        return {"status": "Success"}
        
    @staticmethod
    def failure(fstring, *args, **kwargs):
        return {"status": "Failure", "extra": fstring.format(*args, **kwargs)}

class ToolCallerNode(Runnable):
    def __init__(self, llm):
        super().__init__()
        self._llm = llm.bind_tools(type(self).TOOLS)

    def invoke(self, state, context):
        messages = [SystemMessage(content=type(self).SYSTEM_MESSAGE)] + state["messages"]
    
        while True:
            message = self._llm.invoke(messages)
            
            if message.tool_calls:
                return {"messages": [message]}
                
            if state.get("debug"):
                print("GOT AN INVALID RESPONSE", message)
                
            messages.extend([
                message,
                HumanMessage(content="That was an invalid response. You must make at least one tool call. Please follow your system prompt more closely and try again.")
            ])

    async def ainvoke(self, state, context):
        messages = [SystemMessage(content=type(self).SYSTEM_MESSAGE)] + state["messages"]
    
        while True:
            message = await self._llm.ainvoke(messages)
            
            if message.tool_calls:
                return {"messages": [message]}
                
            if state.get("debug"):
                print("GOT AN INVALID RESPONSE", message)
                
            messages.extend([
                message,
                HumanMessage(content="That was an invalid response. You must make at least one tool call. Please follow your system prompt more closely and try again.")
            ])
