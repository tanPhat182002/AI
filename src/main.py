"""
File chính để chạy ứng dụng Chatbot AI
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời
"""

# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv  # Đọc file .env chứa API key
from seed_data import seed_milvus, seed_milvus_live  # Hàm xử lý dữ liệu
from agent import get_retriever, get_llm_and_agent  # Khởi tạo AI
from langchain.callbacks import StreamlitCallbackHandler  # Hiển thị kết quả realtime
from langchain.memory import StreamlitChatMessageHistory  # Lưu lịch sử chat

# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="AI Assistant",  # Tiêu đề tab trình duyệt
        page_icon="💬",  # Icon tab
        layout="wide"  # Giao diện rộng
    )

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    """
    load_dotenv()  # Đọc API key từ file .env
    setup_page()  # Thiết lập giao diện

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn:
    1. Chọn nguồn dữ liệu (File hoặc URL)
    2. Nhập thông tin file/URL
    3. Nút tải/crawl dữ liệu
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Chọn nguồn dữ liệu
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"]
        )
        
        # Xử lý tùy theo lựa chọn
        if data_source == "File Local":
            handle_local_file()
        else:
            handle_url_input()

def handle_local_file():
    """
    Xử lý khi người dùng chọn tải file:
    1. Nhập tên file và thư mục
    2. Tải dữ liệu khi nhấn nút
    """
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")
    
    if st.button("Tải dữ liệu từ file"):
        with st.spinner("Đang tải dữ liệu..."):
            seed_milvus('http://localhost:19530', 'data_test', filename, directory)
        st.success("Đã tải dữ liệu thành công!")

def handle_url_input():
    """
    Xử lý khi người dùng chọn crawl URL:
    1. Nhập URL cần crawl
    2. Bắt đầu crawl khi nhấn nút
    """
    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    if st.button("Crawl dữ liệu"):
        with st.spinner("Đang crawl dữ liệu..."):
            seed_milvus_live(url, 'http://localhost:19530', 'data_test_live_v2', 'stack-ai')
        st.success("Đã crawl dữ liệu thành công!")

# === GIAO DIỆN CHAT CHÍNH ===
def setup_chat_interface():
    """
    Tạo giao diện chat chính:
    1. Hiển thị tiêu đề
    2. Khởi tạo lịch sử chat
    3. Hiển thị các tin nhắn
    """
    st.title("💬 AI Assistant")
    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và OpenAI")

    # Khởi tạo bộ nhớ chat
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    # Tạo tin nhắn chào mừng nếu là chat mới
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
def handle_user_input(msgs, agent_executor):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Stack AI!"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Gọi AI xử lý
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}
            )

            # Lưu và hiển thị câu trả lời
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

# === HÀM CHÍNH ===
def main():
    """
    Hàm chính điều khiển luồng chương trình:
    1. Khởi tạo ứng dụng
    2. Tạo giao diện
    3. Xử lý tương tác người dùng
    """
    initialize_app()
    setup_sidebar()
    msgs = setup_chat_interface()
    
    # Khởi tạo AI
    retriever = get_retriever()
    agent_executor = get_llm_and_agent(retriever)
    
    # Xử lý chat
    handle_user_input(msgs, agent_executor)

# Chạy ứng dụng
if __name__ == "__main__":
    main() 