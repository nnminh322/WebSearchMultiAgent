from utils.helper import agent_system_prompt
from core.llm_factory import LLM_factory
from langchain.agents import create_agent
from utils.helper import agent_system_prompt
from orchestrations.state import State
from langgraph.types import Command
from typing import Literal
from langchain_core.messages import HumanMessage

CHART_SUMMARIZER_INSTRUCTIONS = open(
    "configs/instructions/chart_summarizer.txt", "r"
).read()


def pr(number: int):
    print(1 + 2)


class ChartSummarizerAgent:
    def __init__(self):
        self.llm = LLM_factory.get_llm("chart_summarizer")
        self.tools = []
        self.system_prompt = agent_system_prompt(suffix=CHART_SUMMARIZER_INSTRUCTIONS)

        self.agent = create_agent(
            model=self.llm, tools=self.tools, system_prompt=self.system_prompt
        )

    def run(self, state: State) -> Command[Literal["executor"]]:
        chart_summarize_results = self.agent.invoke(state)  # type: ignore

        last_msg = chart_summarize_results["messages"][-1]
        last_msg = HumanMessage(
            content=last_msg.content, name="chart_generator"
        )  # re role

        return Command(update={"messages": last_msg}, goto="executor")
