from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain.agents import create_agent
from utils.helper import agent_system_prompt
from core.llm_factory import LLM_factory
from orchestrations.state import State
from typing import Literal
from langgraph.types import Command
from langchain_core.messages import HumanMessage

CHART_GENERATOR_INSTRUCTIONS = open(
    "configs/instructions/chart_generator.txt", "r"
).read()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""

    repl = PythonREPL()
    try:
        exec_run_code = repl.run(code)
        return f"Executed:\n{code}\nStdout: {exec_run_code}"
    except Exception as e:
        return f"Error: {e}"


class ChartGeneratorAgent:
    def __init__(self):
        self.llm = LLM_factory.get_llm("chart_generator")
        self.system_prompt = agent_system_prompt(suffix=CHART_GENERATOR_INSTRUCTIONS)
        self.tools = [python_repl_tool]

        self.agents = create_agent(
            model=self.llm, tools=self.tools, system_prompt=self.system_prompt
        )

    def run(self, state: State) -> Command[Literal["chart_summarizer"]]:
        chart_gen_results = self.agents.invoke(state)  # type: ignore

        last_msg = chart_gen_results["messages"][-1]
        last_msg = HumanMessage(
            content=last_msg.content, name="chart_generator"
        )  # re role

        return Command(update={"messages": last_msg}, goto="chart_summarizer")
