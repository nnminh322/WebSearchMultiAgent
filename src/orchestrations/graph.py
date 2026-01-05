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
    chart_gen = ChartGeneratorAgent()
    chart_sum = ChartSummarizerAgent()
    synthesizer = SynthesizerAgent()

    workflow = StateGraph(State)

    workflow.add_node("planner", planner.run)
    workflow.add_node("executor", executor.run)
    workflow.add_node("web_researcher", web_researcher.run)
    workflow.add_node("chart_gen", chart_gen.run)
    workflow.add_node("chart_sum", chart_sum.run)
    workflow.add_node("synthesizer", synthesizer.run)

    workflow.add_edge(START, "planner")
    
    return workflow.compile()
