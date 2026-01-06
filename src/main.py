import sys
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

sys.path.append(os.getcwd()) 

load_dotenv() 

from orchestrations.graph import build_graph

def main():
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Cảnh báo: Chưa thấy TAVILY_API_KEY. Web Search có thể lỗi.")

    print("🚀 Đang khởi động hệ thống Multi-Agent...")
    graph = build_graph()

    print("\n--- BẮT ĐẦU CHAT (Gõ 'exit' để thoát) ---")
    while True:
        user_input = input("\n👤 Bạn: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Tạm biệt!")
            break
        
        if not user_input.strip():
            continue

        print("🤖 Agent đang suy nghĩ...")
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "enabled_agents": ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer"]
        }

        config = {"recursion_limit": 50}

        try:
            for event in graph.stream(initial_state, config=config): # type: ignore
                for node, values in event.items():
                    print(f"   ⚙️  [Node: {node}] đã chạy xong.")
                    
                    if "final_answer" in values:
                        print(f"\n✅ FINAL ANSWER:\n{values['final_answer']}")
                        print("-" * 50)
                        
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()