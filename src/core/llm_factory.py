import os
import yaml
from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import SecretStr


def load_config():
    with open("configs/agent_config.yaml", "r") as f:
        return yaml.safe_load(f)


GLOBAL_AGENT_CONFIG = load_config()


def get_llm(agent_name: str = "default", temperature: Optional[float] = None):
    default_config = GLOBAL_AGENT_CONFIG.get("default", {})
    specific_config = GLOBAL_AGENT_CONFIG.get(agent_name)

    # PHẢI COPY để không làm hỏng biến toàn cục
    agent_config = (specific_config if specific_config else default_config).copy()

    api_key_name = agent_config.pop("api_key_name")
    api_key = os.getenv(api_key_name)
    api_key = SecretStr(api_key) if api_key else None
    if temperature is not None:
        agent_config["temperature"] = temperature

    try:
        llm = ChatOpenAI(api_key=api_key, **agent_config)
    except Exception as e:
        raise ValueError(f"Error when get llm: {e}")

    return llm
