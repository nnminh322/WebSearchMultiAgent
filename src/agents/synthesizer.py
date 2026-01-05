from core.llm_factory import LLM_factory
from orchestrations.state import State
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.graph import END

SYNTHESIZER_INSTRUCTIONS = open("configs/instructions/synthesizer.txt", "r").read()


class SynthesizerAgent:
    def __init__(self):
        self.llm = LLM_factory.get_llm("synthesizer")
        self.instructions = SYNTHESIZER_INSTRUCTIONS

    def run(self, state: State) -> Command[str]:
        """
        Creates a concise, human‑readable summary of the entire interaction,
        **purely in prose**.

        It ignores structured tables or chart IDs and instead rewrites the
        relevant agent messages (research results, chart commentary, etc.)
        into a short final answer.
        """

        relevant_msgs = [
            str(m.content)
            for m in state.get("messages", [])
            if getattr(m, "name", None)
            in ("web_researcher", "chart_generator", "chart_summarizer")
        ]

        user_question = state.get(
            "user_query",
            state.get("messages", [{}])[0].content if state.get("messages") else "",
        )

        summary_prompt = [
            HumanMessage(
                content=(
                    f"User question: {user_question}\n\n"
                    f"{self.instructions}\n\n"
                    f"Context:\n\n" + "\n\n---\n\n".join(relevant_msgs)
                )
            )
        ]

        response = self.llm.invoke(summary_prompt)
        answer = str(response.content).strip()

        return Command(
            update={
                "final_answer": answer,
                "messages": [HumanMessage(content=answer, name="synthesizer")],
            },
            goto=END,
        )
