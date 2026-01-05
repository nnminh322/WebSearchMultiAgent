from trulens.apps.langgraph import TruGraph
from src.evaluation.metrics import AgentMetrics

def create_tru_recorder(graph, app_name="Web Search Agent", app_version="v1.0"):
    """
    Wrap LangGraph bằng TruGraph Recorder với các metrics đã định nghĩa.
    """
    metrics = AgentMetrics()
    feedbacks = metrics.get_all_feedbacks()

    tru_recorder = TruGraph(
        graph,
        app_name=app_name,
        app_version=app_version,
        feedbacks=feedbacks
    )
    return tru_recorder