#!/usr/bin/env python3

import configparser
import os
import sys
import readline

from workflows.react import react_graph
from workflows.pax import pax_graph

from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_model(cfgfile):
    cfg = configparser.ConfigParser()
    cfg.read(cfgfile)
    
    for name, section in cfg.items():
        temperature = section.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
            
        try:
            if section.get("backend") == "ollama":
                model = ChatOllama(
                    model = section.get("model", "llama3.1:8b").strip(),
                    base_url = section.get("base_url", "http://localhost:11434").strip(),
                    temperature = temperature
                )
                    
            elif section.get("backend") == "openai":
                model = ChatOpenAI(
                    model = section.get("model", "gpt-5-mini-2025-08-07").strip(),
                    api_key = section["api_key"].strip(),
                    base_url = section.get("base_url", "https://api.openai.com/v1").strip(),
                    temperature = temperature
                )
            
            # Test prompt intended to not consume very many tokens
            model.invoke([HumanMessage(content="Say \"aardwolf\"")])
        except Exception:
            pass # Intentional fallthrough
        else:
            return model

if __name__ == "__main__":
    histfile = ".llm_history"
    cfgfile = "config.ini"
    
    workflows = {
        "react": react_graph,
        "pax": pax_graph
    }
    
    if len(sys.argv) < 2:
        print("Usage: {} <{}>".format(sys.argv[0], "|".join(workflows.keys())))
        sys.exit(-1)
    
    graphfunc = workflows.get(sys.argv[1].lower())
    
    if graphfunc is None:
        print("Usage: {} <{}>".format(sys.argv[0], "|".join(workflows.keys())))
        sys.exit(-1)
    
    graph, conversation = graphfunc(get_model(cfgfile))
    
    if os.path.exists(histfile):
        readline.read_history_file(histfile)
        
    try:
        while True:
            try:
                content = input("> ")
                conversation["messages"].append(HumanMessage(content=content))
                
                for step in graph.stream(conversation, stream_mode="values"):
                    step["messages"][-1].pretty_print()
                    conversation = step
            except (EOFError, KeyboardInterrupt):
                print()
                break
    finally:
        readline.write_history_file(histfile)
