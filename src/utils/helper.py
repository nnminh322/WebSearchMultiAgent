from typing import Any, Dict, List
from orchestrations.state import State


def get_enabled_agents(state: Any) -> List[str]:
    baseline = ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer"]
    if not state:
        return baseline
    val = (
        state.get("enabled_agents")
        if isinstance(state, dict)
        else getattr(state, "enabled_agents", None)
    )

    if isinstance(val, List) and val:
        allowed = set(baseline)
        filtered = [a for a in val if a in allowed]
        return filtered
    return baseline


def agent_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )
