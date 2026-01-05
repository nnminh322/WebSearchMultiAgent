from core.llm_factory import LLM_factory
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langgraph.types import Command
from typing import Literal
from orchestrations.state import State
from langchain_core.messages import HumanMessage
from utils.helper import agent_system_prompt

WEB_RESEARCHER_PROMPT_TEMPLATE = open("configs/prompts/web_researcher.txt", "r").read()


class WebResearcherAgent:
    def __init__(self):
        self.llm = LLM_factory.get_llm("web_reseacher")
        self.tools = [TavilySearch(max_results=5)]
        self.system_prompt = agent_system_prompt(suffix=WEB_RESEARCHER_PROMPT_TEMPLATE)
        
        self.agents = create_agent(
            model=self.llm, tools=self.tools, system_prompt=self.system_prompt
        )

    def run(self, state: State) -> Command[Literal["executor"]]:
        agent_query = state.get("agent_query")

        web_search_results = self.agents.invoke(
            {"messages": [HumanMessage(content=agent_query)]}
        )
        last_msg = web_search_results["messages"][-1]
        last_msg = HumanMessage(
            content=last_msg.content, name="web_researcher"
        )  # re role

        return Command(update={"messages": [last_msg]}, goto="executor")
