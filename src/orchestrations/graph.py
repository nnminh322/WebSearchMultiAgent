from langgraph.graph import START, END, StateGraph
from orchestrations.state import State
from agents.planner import PlannerAgent
from agents.web_researcher import WebResearcherAgent
from agents.chart_generator import ChartGeneratorAgent
from agents.chart_summarizer import ChartSummarizerAgent
from agents.synthesizer import SynthesizerAgent
from agents.executor import ExecutorAgent


def build_graph():

    planner = PlannerAgent()
    executor = ExecutorAgent()
    web_researcher = WebResearcherAgent()
    chart_generator = ChartGeneratorAgent()
    chart_summarizer = ChartSummarizerAgent()
    synthesizer = SynthesizerAgent()

    workflow = StateGraph(State)

    workflow.add_node("planner", planner.run)
    workflow.add_node("executor", executor.run)
    workflow.add_node("web_researcher", web_researcher.run)
    workflow.add_node("chart_generator", chart_generator.run)
    workflow.add_node("chart_summarizer", chart_summarizer.run)
    workflow.add_node("synthesizer", synthesizer.run)

    workflow.add_edge(START, "planner")
    
    return workflow.compile()
