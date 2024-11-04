# Import các thư viện cần thiết
from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_openai import ChatOpenAI  # Model ngôn ngữ OpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent  # Tạo và thực thi agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt
from seed_data import seed_milvus, connect_to_milvus  # Kết nối với Milvus
from langchain.callbacks import StreamlitCallbackHandler  # Xử lý callback cho Streamlit
from langchain.memory import StreamlitChatMessageHistory  # Lưu trữ lịch sử chat
from langchain.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever  # Retriever dựa trên BM25
from langchain_core.documents import Document  # Lớp Document

def get_retriever() -> EnsembleRetriever:
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Returns:
        EnsembleRetriever: Retriever kết hợp với tỷ trọng:
            - 70% Milvus vector search (k=4 kết quả)
            - 30% BM25 text search (k=4 kết quả)
    Chú ý:
        - Yêu cầu Milvus server đang chạy tại localhost:19530
        - Collection 'data_test_live_v2' phải tồn tại trong Milvus
        - BM25 được khởi tạo từ 100 document đầu tiên trong Milvus
    """
    # Kết nối với Milvus và tạo vector retriever
    vectorstore = connect_to_milvus('http://localhost:19530', 'data_test_live_v2')
    milvus_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Tạo BM25 retriever từ toàn bộ documents
    documents = [Document(page_content=doc.page_content, metadata=doc.metadata)
                 for doc in vectorstore.similarity_search("", k=100)]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    # Kết hợp hai retriever với tỷ trọng
    ensemble_retriever = EnsembleRetriever(
        retrievers=[milvus_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever

# Tạo công cụ tìm kiếm cho agent
tool = create_retriever_tool(
    get_retriever(),
    "find",
    "Search for information of Stack AI."
)

def get_llm_and_agent(_retriever) -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể
    Args:
        _retriever: Retriever đã được cấu hình để tìm kiếm thông tin
    Returns:
        AgentExecutor: Agent đã được cấu hình với:
            - Model: GPT-4
            - Temperature: 0
            - Streaming: Enabled
            - Custom system prompt
    Chú ý:
        - Yêu cầu OPENAI_API_KEY đã được cấu hình
        - Agent được thiết lập với tên "ChatchatAI"
        - Sử dụng chat history để duy trì ngữ cảnh hội thoại
    """
    # Khởi tạo ChatOpenAI với chế độ streaming
    llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
    tools = [tool]
    
    # Thiết lập prompt template cho agent
    system = """You are an expert at AI. Your name is ChatchatAI."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tạo và trả về agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Khởi tạo retriever và agent
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)