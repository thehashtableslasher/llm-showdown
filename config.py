import configparser

from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_model(cfgfile = "config.ini"):
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
            
    return None
