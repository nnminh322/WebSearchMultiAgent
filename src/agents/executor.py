import json
import yaml
from typing import Literal, Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from orchestrations.state import State
from core.llm_factory import LLM_factory
from utils.helper import get_enabled_agents

EXECUTOR_PROMPT_TEMPLATE = open("configs/prompts_template/executor.txt").read()

with open("configs/graph_config.yaml", "r") as f:
    graph_config = yaml.safe_load(f)
    MAX_REPLANS = graph_config["system"]["max_replans"]

with open("configs/agents_description.yaml", "r") as f:
    AGENTS_DESCRIPTIONS = yaml.safe_load(f)


def format_agent_guidelines_for_executor(state: State | None = None) -> str:
    agents_des = AGENTS_DESCRIPTIONS
    enabled_agents = get_enabled_agents(state=state)
    executor_guidelines = []

    if "web_researcher" in enabled_agents:
        web_desc = agents_des["web_researcher"]
        executor_guidelines.append(
            f"- Use `\"web_researcher\"` when {web_desc['use_when'].lower()}."
        )
    if "cortex_researcher" in enabled_agents:
        cortex_desc = agents_des["cortex_researcher"]
        executor_guidelines.append(
            f"- Use `\"cortex_researcher\"` for {cortex_desc['use_when'].lower()}."
        )

    return "\n".join(executor_guidelines)


class ExecutorAgent:
    def __init__(self):
        self.llm = LLM_factory.get_llm("executor")
        self.max_replans = MAX_REPLANS

    def run(
        self, state: State
    ) -> Command[
        Literal["web_researcher", "chart_generator", "synthesizer", "planner"]
    ]:  # type: ignore

        plan = state.get("plan", {})
        step = state.get("current_step", 1)
        latest_plan: Dict[str, Any] = state.get("plan") or {}  # type: ignore
        plan_block: Dict[str, Any] = latest_plan.get(str(step), {})
        plan_agent = plan_block.get("agent", "web_researcher")
        executor_guidelines = format_agent_guidelines_for_executor(state=state)
        messages_tail = (state.get("messages") or [])[-4:]
        user_query = state.get("user_query", "")
        replan_flag = state.get("replan_flag")
        if state.get("replan_flag"):
            planned_agent = plan.get(str(step), {}).get("agent")  # type: ignore
            return Command(
                update={"replan_flag": False, "current_step": step + 1},
                goto=planned_agent,
            )
        type_executors = "`, `".join(
            sorted(
                set(
                    [
                        a
                        for a in get_enabled_agents(state)
                        if a
                        in [
                            "web_researcher",
                            "chart_generator",
                            "chart_summarizer",
                            "synthesizer",
                        ]
                    ]
                    + ["planner"]
                )
            )
        )
        goto = "|".join(
            [
                a
                for a in get_enabled_agents(state)
                if a
                in [
                    "web_researcher",
                    "chart_generator",
                    "chart_summarizer",
                    "synthesizer",
                ]
            ]
            + ["planner"]
        )
        prompt_content = EXECUTOR_PROMPT_TEMPLATE.format(
            type_executors=type_executors,
            executor_guidelines=executor_guidelines,
            MAX_REPLANS=MAX_REPLANS,
            max_replans=self.max_replans,
            goto=goto,
            plan_agent=plan_agent,
            user_query=user_query,
            step=step,
            plan_block=plan_block,
            replan_flag=replan_flag,
            messages_tail=messages_tail,
        )

        response = self.llm.invoke(HumanMessage(content=prompt_content))  # type: ignore
        try:
            content_str = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            parsed = json.loads(content_str)
            replan: bool = parsed["replan"]
            goto: str = parsed["goto"]
            reason: str = parsed["reason"]
            query: str = parsed["query"]
        except Exception as exc:
            raise ValueError(f"Invalid executor JSON:\n{response.content}") from exc

        updates = {
            "messages": [HumanMessage(content=response.content, name="executor")],
            "last_reason": reason,
            "agent_query": query,
        }

        replan_attempts: Dict[int, int] = state.get("replan_attempts", {}) or {}
        step_replan = replan_attempts.get(step, 0)  

        if replan:
            if step_replan < MAX_REPLANS:
                replan_attempts[step] = step_replan + 1 
                updates.update(
                    {
                        "replan_attempts": replan_attempts,
                        "replan_flag": True,
                        "current_step": step,
                    }
                )
                return Command(update=updates, goto="planner")
            else:
                next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer") # type: ignore
                updates["current_step"] = step + 1
                return Command(update=updates, goto=next_agent)

        planned_agent = plan.get(str(step), {}).get("agent") # type: ignore
        updates["current_step"] = step + 1 if goto == planned_agent else step
        updates["replan_flag"] = False
        return Command(update=updates, goto=goto) # type: ignore
        
