# main.py
import streamlit as st
import logging
from typing import Optional
from dotenv import load_dotenv
from seed_data import seed_milvus, seed_milvus_live
from agent import get_retriever as get_openai_retriever, get_llm_and_agent as get_openai_agent
from local_ollama import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread safety for session state
session_lock = Lock()

class ChatInterface:
    """Enhanced chat interface management"""
    def __init__(self):
        self.msgs = StreamlitChatMessageHistory(key="langchain_messages")
        
    def initialize_chat(self):
        """Initialize chat with welcome message"""
        with session_lock:
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
                ]
                self.msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")
                
    def display_messages(self):
        """Display chat messages"""
        for msg in st.session_state.messages:
            role = "assistant" if msg["role"] == "assistant" else "human"
            st.chat_message(role).write(msg["content"])
            
    def handle_user_input(self, agent_executor):
        """Process user input with error handling"""
        if prompt := st.chat_input("Hãy hỏi tôi điều gì đó..."):
            try:
                # Display user message
                with session_lock:
                    st.session_state.messages.append({"role": "human", "content": prompt})
                st.chat_message("human").write(prompt)
                self.msgs.add_user_message(prompt)

                # Process and display response
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Đang tìm kiếm thông tin..."):
                            # Setup callback
                            st_callback = StreamlitCallbackHandler(st.container())
                            
                            # Get chat history
                            chat_history = [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in st.session_state.messages[:-1]
                            ]

                            # Get AI response
                            response = agent_executor.invoke(
                                {
                                    "input": prompt,
                                    "chat_history": chat_history
                                },
                                {"callbacks": [st_callback]}
                            )
                            
                            output = response["output"]
                            
                            # Update messages
                            with session_lock:
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": output}
                                )
                            self.msgs.add_ai_message(output)
                            st.write(output)
                            
                    except Exception as e:
                        error_msg = "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."
                        st.error(error_msg)
                        logger.error(f"Agent execution error: {str(e)}")
                        
            except Exception as e:
                st.error("Có lỗi xảy ra. Vui lòng thử lại.")
                logger.error(f"Message handling error: {str(e)}")

class SidebarManager:
    """Enhanced sidebar management"""
    def setup_sidebar(self) -> str:
        """Setup sidebar with improved UI/UX"""
        with st.sidebar:
            st.title("⚙️ Cấu hình")
            
            # Embeddings Model Selection
            st.header("🔤 Embeddings Model")
            embeddings_choice = st.radio(
                "Chọn Embeddings Model:",
                ["OpenAI", "Ollama"],
                help="Chọn model để tạo embeddings cho documents"
            )
            use_ollama_embeddings = (embeddings_choice == "Ollama")
            
            # Data Source Configuration
            st.header("📚 Nguồn dữ liệu")
            data_source = st.radio(
                "Chọn nguồn dữ liệu:",
                ["File Local", "URL trực tiếp"]
            )
            
            # Handle data source
            if data_source == "File Local":
                self._handle_local_file(use_ollama_embeddings)
            else:
                self._handle_url_input(use_ollama_embeddings)
            
            # AI Model Selection
            st.header("🤖 Model AI")
            model_choice = st.radio(
                "Chọn AI Model:",
                ["GitHub AI", "Ollama (Local)"],
                help="Chọn model AI để xử lý câu hỏi"
            )
            
            return model_choice
            
    def _handle_local_file(self, use_ollama_embeddings: bool):
        """Handle local file input"""
        collection_name = st.text_input(
            "Tên collection:",
            "data_test",
            help="Nhập tên collection trong Milvus"
        )
        
        filename = st.text_input(
            "Tên file JSON:",
            "stack.json",
            help="Nhập tên file JSON chứa dữ liệu"
        )
        
        directory = st.text_input(
            "Thư mục chứa file:",
            "data",
            help="Nhập đường dẫn đến thư mục chứa file"
        )
        
        if st.button("Tải dữ liệu từ file", help="Click để bắt đầu tải dữ liệu"):
            self._process_local_file(
                collection_name,
                filename,
                directory,
                use_ollama_embeddings
            )
            
    def _process_local_file(
        self,
        collection_name: str,
        filename: str,
        directory: str,
        use_ollama: bool
    ):
        """Process local file loading"""
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        try:
            with st.spinner("Đang tải dữ liệu..."):
                seed_milvus(
                    'http://localhost:19530',
                    collection_name,
                    filename,
                    directory,
                    use_ollama=use_ollama
                )
                st.success(f"Đã tải dữ liệu vào collection '{collection_name}'!")
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            logger.error(f"Local file processing error: {str(e)}")
            
    def _handle_url_input(self, use_ollama_embeddings: bool):
        """Handle URL input"""
        collection_name = st.text_input(
            "Tên collection:",
            "data_test_live",
            help="Nhập tên collection trong Milvus"
        )
        
        url = st.text_input(
            "Nhập URL:",
            "https://www.stack-ai.com/docs",
            help="Nhập URL để crawl dữ liệu"
        )
        
        if st.button("Crawl dữ liệu", help="Click để bắt đầu crawl"):
            self._process_url(collection_name, url, use_ollama_embeddings)
            
    def _process_url(self, collection_name: str, url: str, use_ollama: bool):
        """Process URL crawling"""
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        try:
            with st.spinner("Đang crawl dữ liệu..."):
                seed_milvus_live(
                    url,
                    'http://localhost:19530',
                    collection_name,
                    'stack-ai',
                    use_ollama=use_ollama
                )
                st.success(f"Đã crawl dữ liệu vào collection '{collection_name}'!")
        except Exception as e:
            st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")
            logger.error(f"URL processing error: {str(e)}")

class App:
    """Main application class"""
    def __init__(self):
        self.chat_interface = ChatInterface()
        self.sidebar_manager = SidebarManager()
        
    def setup_page(self):
        """Configure page settings"""
        st.set_page_config(
            page_title="AI Assistant",
            page_icon="💬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize(self):
        """Initialize application"""
        load_dotenv()
        self.setup_page()
        
    def setup_chat_interface(self, model_choice: str):
        """Setup main chat interface"""
        st.title("💬 AI Assistant")
        
        # Dynamic caption based on model
        caption = ("🚀 Trợ lý AI được hỗ trợ bởi LangChain và " + 
                  ("GitHub AI" if model_choice == "GitHub AI" else "Ollama LLaMA2"))
        st.caption(caption)
        
        # Initialize chat
        self.chat_interface.initialize_chat()
        self.chat_interface.display_messages()
        
    def run(self):
        """Run application"""
        try:
            # Initialize app
            self.initialize()
            
            # Setup sidebar and get model choice
            model_choice = self.sidebar_manager.setup_sidebar()
            
            # Setup chat interface
            self.setup_chat_interface(model_choice)
            
            # Initialize AI model
            retriever = (get_openai_retriever() if model_choice == "GitHub AI"
                        else get_ollama_retriever())
            agent_executor = (get_openai_agent(retriever) if model_choice == "GitHub AI"
                            else get_ollama_agent(retriever))
            
            # Handle user input
            self.chat_interface.handle_user_input(agent_executor)
            
        except Exception as e:
            st.error("Ứng dụng gặp lỗi. Vui lòng thử lại sau.")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    app = App()
    app.run() 
