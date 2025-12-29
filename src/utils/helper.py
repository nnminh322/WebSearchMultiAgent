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
