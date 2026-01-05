import json
import yaml
from typing import Dict, Any, Literal
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from orchestrations.state import State
from utils.helper import get_enabled_agents
from core.llm_factory import LLM_factory


class PlannerAgent:
    def __init__(self):
        self.llm = LLM_factory.get_llm("planner")

    def _load_configs(self):
        """Load all necessary yaml and text configurations."""
        with open("configs/agents_description.yaml", "r", encoding="utf-8") as f:
            self.agents_descriptions = yaml.safe_load(f)

        with open("configs/agent_config.yaml", "r", encoding="utf-8") as f:
            agent_config = yaml.safe_load(f)
            self.planner_config = agent_config.get("planner", {})

        with open("configs/graph_config.yaml", "r", encoding="utf-8") as f:
            graph_config = yaml.safe_load(f)
            self.max_replans = graph_config.get("system", {}).get("max_replans", 3)

        with open("configs/prompts_template/planner.txt", "r", encoding="utf-8") as f:
            self.plan_prompt_template = f.read()

        with open("configs/prompts_template/replan.txt", "r", encoding="utf-8") as f:
            self.replan_prompt_template = f.read()

    def _format_agent_list(self, state: State) -> str:
        enabled_list = get_enabled_agents(state=state)
        agent_list = []
        for agent_key, agent_detail in self.agents_descriptions.items():
            if agent_key in enabled_list:
                agent_list.append(f"  • '{agent_key}' - {agent_detail['capability']}")

        return "\n".join(agent_list)

    def _format_guidelines(self, state: State) -> str:
        enabled_set = set(get_enabled_agents(state=state))
        guidelines = []

        if "web_researcher" in enabled_set:
            web_search_desc = self.agents_descriptions["web_researcher"]
            guidelines.append(
                f"- Use 'web_researcher' for {web_search_desc['use_when'].lower()}."
            )

        if "chart_generator" in enabled_set:
            chart_gen_desc = self.agents_descriptions["chart_generator"]
            cs_hint = (
                " A 'chart_summarizer' should be used to summarize the chart."
                if "chart_summarizer" in enabled_set
                else ""
            )

            guidelines.append(
                f"- **Include `chart_generator` _only_ if {chart_gen_desc['use_when'].lower()}**. "
                f"If included, `chart_generator` must be {chart_gen_desc['position_requirement'].lower()}. "
                f"Visualizations should include all of the data from the previous steps that is reasonable for the chart type.{cs_hint}"
            )

        if "synthesizer" in enabled_set:
            synth_desc = self.agents_descriptions["synthesizer"]
            guidelines.append(
                f"  - Otherwise use `synthesizer` as {synth_desc['position_requirement'].lower()}, "
                f"and be sure to include all of the data from the previous steps."
            )

        return "\n".join(guidelines)

    def _build_plan_prompt(self, state: State) -> HumanMessage:
        """
        Build the prompt that instructs the LLM to return a high-level plan.
        """
        replan_flag = state.get("replan_flag", False)
        user_query = state.get("user_query", "")
        prior_plan = state.get("plan") or {}
        replan_reason = state.get("last_reason", "")

        agent_list_str = self._format_agent_list(state=state)
        guidelines_str = self._format_guidelines(state=state)
        enabled_list = get_enabled_agents(state=state)

        planner_agent_enum = " | ".join(enabled_list)

        base_instruction = self.plan_prompt_template.format(
            agent_list_str=agent_list_str,
            planner_agent_enum=planner_agent_enum,
            guidelines_str=guidelines_str,
        )

        if replan_flag:
            current_plan = json.dumps(prior_plan, indent=2)
            replan_instruction = self.replan_prompt_template.format(
                replan_reason=replan_reason, current_plan=current_plan
            )
            prompt_content = base_instruction + replan_instruction
        else:
            prompt_content = base_instruction + "\nGenerate a new plan from scratch."

        prompt_content += f"\nUser_query: {user_query}"

        return HumanMessage(content=prompt_content)

    def run(self, state: State) -> Command:
        """
        Runs the planning LLM and stores the resulting plan in state.
        """
        prompt = self._build_plan_prompt(state=state)
        llm_reply = self.llm.invoke([prompt])

        try:
            response_content = (
                llm_reply.content
                if isinstance(llm_reply.content, str)
                else str(llm_reply.content)
            )
            parsed_plan = json.loads(response_content)
        except json.JSONDecodeError:
            raise ValueError(f"Planner returned invalid JSON:\n{llm_reply.content}")

        replan_flag = state.get("replan_flag", False)
        updated_plan: Dict[str, Any] = parsed_plan

        return Command(
            update={
                "plan": updated_plan,
                # Fix: Đổi key "message" thành "messages" để đúng chuẩn LangGraph state
                "messages": [
                    HumanMessage(
                        content=llm_reply.content,
                        name="replan" if replan_flag else "initial_plan",
                    )
                ],
                "user_query": state.get("user_query", state["messages"][0].content),
                "current_step": 1 if not replan_flag else state["current_step"],
                "replan_flag": state.get("replan_flag", False),
                "last_reason": "",
                "enabled_agents": state.get("enabled_agents"),
            },
            goto="executor",
        )
