from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import MessagesState

class State(MessagesState):
    user_query: Optional[str]
    enable_agent: Optional[List[str]]
    plan: Optional[List[Dict[int, Dict[str, Any]]]]
    current_step: int
    agent_query: Optional[str]
    last_reason: Optional[str]
    replan_flag: Optional[bool]
    replan_attempts: Optional[Dict[int, int]]
