import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from trulens.dashboard import run_dashboard
from src.evaluation.config import get_trulens_session

def main():
    print("📊 Launching TruLens Dashboard...")
    # Khởi tạo session để đảm bảo kết nối DB đúng
    _ = get_trulens_session() 
    
    # Chạy dashboard (block process)
    run_dashboard(port=8501) # Mặc định chạy port 8501

if __name__ == "__main__":
    main()