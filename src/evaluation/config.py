import os
from trulens.core.database.connector.default import DefaultDBConnector
from trulens.core.session import TruSession

def get_trulens_session(db_name: str = "data_agent.sqlite"):
    """Khởi tạo TruLens Session và Connector."""
    
    # Xóa file cũ nếu muốn reset mỗi lần chạy test (Optional)
    # if os.path.exists(db_name):
    #     os.remove(db_name)

    database_url = f"sqlite:///{db_name}"
    connector = DefaultDBConnector(database_url=database_url)
    session = TruSession(connector=connector)
    return session

def reset_eval_db(session: TruSession):
    """Xóa sạch dữ liệu cũ để test mới."""
    session.reset_database()