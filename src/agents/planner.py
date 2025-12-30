import json
import yaml
from typing import Dict, Any, Literal
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from orchestrations.state import State
from utils.helper import get_enabled_agents
from core.llm_factory import get_llm

with open("configs/agents_description.yaml", "r") as f:
    AGENTS_DESCRIPTIONS = yaml.safe_load(f)

with open("configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)
    planner_config = agent_config["planner"]

with open("configs/graph_config/yaml", "r") as f:
    graph_config = yaml.safe_load(f)
    MAX_REPLANS = graph_config["system"]["max_replans"]


PLAN_PROMPT_TEMPLATE = open("configs/prompts/planner.txt", "r").read()
REPLAN_PROMPT_TEMPLATE = open("configs/prompts/replan.txt", "r").read()
REASONING_LLM = get_llm("planner")



def format_agent_list(state: State) -> str:
    enabled_list = get_enabled_agents(state=state)
    agent_list = []
    for agent_key, agent_detail in AGENTS_DESCRIPTIONS.item():
        if agent_key in enabled_list:
            agent_list.append(f"  • '{agent_key}' - {agent_detail['capability']}")

    return "\n".join(agent_list)


def format_guidelines(state: State) -> str:
    enabled_set = set(get_enabled_agents(state=state))
    guidelines = []

    if "web_researcher" in enabled_set:
        web_search_desc = AGENTS_DESCRIPTIONS["web_researcher"]
        guidelines.append(
            f"- Use 'web_researcher' for {web_search_desc['use_when'].lower()}."
        )

    if "chart_generator" in enabled_set:
        chart_gen_desc = AGENTS_DESCRIPTIONS["chart_generator"]
        cs_hint = (
            " A 'chart_summarizer' should be used to summarize the chart."
            if "chart_summarizer" in enabled_set
            else ""
        )

        guidelines.append(
            f"- **Include `chart_generator` _only_ if {chart_gen_desc['use_when'].lower()}**. If included, `chart_generator` must be {chart_gen_desc['position_requirement'].lower()}. Visualizations should include all of the data from the previous steps that is reasonable for the chart type.{cs_hint}"
        )

    if "synthesizer" in enabled_set:
        synth_desc = AGENTS_DESCRIPTIONS["synthesizer"]
        guidelines.append(
            f"  - Otherwise use `synthesizer` as {synth_desc['position_requirement'].lower()}, and be sure to include all of the data from the previous steps."
        )

    return "\n".join(guidelines)


def plan_prompt(state: State) -> HumanMessage:
    """
    Build the prompt that instructs the LLM to return a high-level plan.
    """
    replan_flag = state.get("replan_flag", False)
    user_query = state.get("user_query", "")
    prior_plan = state.get("plan") or {}
    replan_reason = state.get("last_reason", "")

    agent_list_str = format_agent_list(state=state)
    guidelines_str = format_guidelines(state=state)
    enabled_list = get_enabled_agents(state=state)

    planner_agent_enum = " | ".join(enabled_list)

    base_instruction = PLAN_PROMPT_TEMPLATE.format(
        agent_list_str=agent_list_str,
        planner_agent_enum=planner_agent_enum,
        guidelines_str=guidelines_str,
    )

    if replan_flag:
        REPLAN_PROMPT_TEMPLATE = open("configs/prompts/replan.txt", "r").read()
        replan_instruction = REPLAN_PROMPT_TEMPLATE.format(
            replan_reason=replan_reason, prior_plan=prior_plan
        )
        prompt_content = base_instruction + replan_instruction
    else:
        prompt_content = base_instruction + "\nGenerate a new plan from scratch."

    prompt_content += f"\nUser_query: {user_query}"

    return HumanMessage(content=prompt_content)



# def planner_node(state: State) -> Command[Literal["executor"]]:
def planner_node(state: State):
    """
    Runs the planning LLM and stores the resulting plan in state.
    Refactored for robustness.
    """
    
    pass