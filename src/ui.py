import os
import sys
import glob
import time
import traceback
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()

from orchestrations.graph import build_graph

st.set_page_config(page_title="Web Search Agent", page_icon="🕵️", layout="wide")
st.title("🕵️ Web Search Multi-Agent System")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input("Nhập câu hỏi nghiên cứu... (VD: Giá vàng hôm nay?)"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        status_container = st.status("🚀 Agent đang khởi động...", expanded=True)
        
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "user_query": prompt,
            "enabled_agents": ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer"]
        }
        
        try:
            existing_pngs = set(glob.glob('*.png'))

            for event in st.session_state.graph.stream(initial_state, config={"recursion_limit": 50}): #type: ignore
                for node, values in event.items():
                    status_container.write(f"✅ **{node}** đã hoàn thành tác vụ.")
                    
                    if node == "web_researcher":
                        status_container.write("🔎 Đang tìm kiếm dữ liệu...")
                    elif node == "chart_generator":
                        status_container.write("📊 Đang vẽ biểu đồ...")
                    
                    if "final_answer" in values:
                        full_response = values["final_answer"]

            status_container.update(label="✅ Hoàn tất!", state="complete", expanded=False)
            
            if full_response:
                message_placeholder.markdown(full_response)
                st.session_state.messages.append(AIMessage(content=full_response))
            else:
                st.warning("Agent không trả về kết quả dạng text (kiểm tra lại log).")

            current_pngs = set(glob.glob('*.png'))
            list_pngs = list(current_pngs)
            if list_pngs:
                latest_file = max(list_pngs, key=os.path.getmtime)
                if time.time() - os.path.getmtime(latest_file) < 60:
                    st.image(latest_file, caption=f"Biểu đồ: {latest_file}")
                    # st.session_state.messages.append(AIMessage(content=f"IMAGE_PATH:{latest_file}"))

        except Exception as e:
            status_container.update(label="❌ Có lỗi xảy ra!", state="error", expanded=True)
            st.error(f"Lỗi: {e}")
            print("\n❌ TRACEBACK CHI TIẾT:")
            traceback.print_exc()