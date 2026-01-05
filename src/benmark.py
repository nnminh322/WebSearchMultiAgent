import sys
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load môi trường và path
load_dotenv()
sys.path.append(os.path.join(os.getcwd(), "src"))

from orchestrations.graph import build_graph
from evaluation.config import get_trulens_session, reset_eval_db
from evaluation.wrapper import create_tru_recorder


def run_benchmark():
    # 1. Setup Session
    print("🛠️  Setting up TruLens Session...")
    session = get_trulens_session()
    # reset_eval_db(session) # Uncomment nếu muốn xóa dữ liệu cũ

    # 2. Build Graph
    print("🚀 Building Agent Graph...")
    graph = build_graph()

    # 3. Create Recorder
    recorder = create_tru_recorder(graph)

    # 4. Define Test Cases
    test_queries = [
        "Chart the current market capitalization of the top 5 banks in the US?",
        "Identify current regulatory changes for the financial services industry in the US.",
    ]

    # 5. Execute & Record
    print(f"🏃 Starting Benchmark with {len(test_queries)} queries...")

    for query in test_queries:
        print(f"\n🧪 Processing Query: {query}")

        state = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "enabled_agents": [
                "web_researcher",
                "chart_generator",
                "chart_summarizer",
                "synthesizer",
            ],
        }

        # Context manager 'with recorder' sẽ tự động capture trace
        with recorder as recording:
            try:
                graph.invoke(state)  # type: ignore
                print("   ✅ Completed successfully.")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

    # 6. Show summary
    print("\n📊 Leaderboard (Top Results):")
    print(session.get_leaderboard())
    print("\n✅ Benchmark Complete. Run 'python dashboard.py' to view details.")


if __name__ == "__main__":
    run_benchmark()
