# agent.py
import os
from functools import lru_cache
from typing import Optional, Dict, List
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from seed_data import GitHubEmbeddings, connect_to_milvus
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubAIChat(ChatOpenAI):
    """Enhanced GitHub AI Chat Model with error handling and caching"""
    def __init__(self, **kwargs):
        self._validate_token()
        super().__init__(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.getenv("GITHUB_TOKEN"),
            model="gpt-4o-mini",
            streaming=True,
            temperature=0,
            request_timeout=30,  # Added timeout
            max_retries=3        # Added retries
        )
    
    def _validate_token(self):
        """Validate GitHub token exists"""
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is not set")

class CustomRetriever:
    """Improved retriever with caching and error handling"""
    def __init__(self, collection_name: str = "data_test"):
        self.collection_name = collection_name
        self.default_doc = self._create_default_doc()
        # Cache for vector store connection
        self._vectorstore = None
        
    @staticmethod
    def _create_default_doc() -> Document:
        """Create default response document"""
        return Document(
            page_content="""
            Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về Stack AI và các chủ đề liên quan đến AI, Machine Learning. 
            Đối với các câu hỏi khác, bạn nên tham khảo các trang web chuyên biệt.
            """.strip(),
            metadata={"source": "default", "type": "system_message"}
        )
    
    @lru_cache(maxsize=100)
    def _get_cached_results(self, query: str) -> List[Document]:
        """Cache search results for identical queries"""
        return self._search_documents(query)
        
    def _search_documents(self, query: str) -> List[Document]:
        """Perform actual document search"""
        try:
            if not self._vectorstore:
                embeddings = GitHubEmbeddings()
                self._vectorstore = connect_to_milvus(
                    'http://localhost:19530',
                    self.collection_name,
                    embeddings=embeddings
                )
            
            results = self._vectorstore.similarity_search(query, k=4)
            if not results:
                logger.warning(f"No documents found for query: {query}")
                return [self.default_doc]
                
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return [self.default_doc]

    def invoke(self, query: str, **kwargs) -> List[Document]:
        """Main retrieval method with caching"""
        return self._get_cached_results(query)
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Alias for invoke method"""
        return self.invoke(query)

class AgentManager:
    """Manager class for agent initialization and execution"""
    def __init__(self, collection_name: str = "data_test"):
        self.collection_name = collection_name
        self.retriever = None
        self.agent_executor = None
        self._initialize()
        
    def _initialize(self):
        """Initialize agent components"""
        try:
            self.retriever = CustomRetriever(self.collection_name)
            self.agent_executor = self._create_agent()
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise
            
    def _create_agent(self) -> AgentExecutor:
        """Create agent with enhanced prompt"""
        try:
            llm = GitHubAIChat()
            
            tool = create_retriever_tool(
                self.retriever,
                "find",
                "Tìm kiếm thông tin trong cơ sở dữ liệu về AI và Stack AI. Chỉ sử dụng tool này cho các câu hỏi liên quan đến AI, Machine Learning và Stack AI."
            )
            
            # Enhanced system prompt
            system_prompt = """
            Bạn là một AI Assistant chuyên về AI và Machine Learning. 
            Tên của bạn là Phát Trần AI.
            
            Hướng dẫn:
            1. Trả lời dựa trên thông tin trong documents nếu có
            2. Nếu không tìm thấy thông tin phù hợp, hãy nói rõ điều đó
            3. Không đưa ra thông tin sai lệch hoặc không chắc chắn
4. Trả lời bằng tiếng anh thôi
5.Hiển thị thông chi tiết sản phẩm và hình ảnh.
6. Gợi ý 3 sản phẩm liên quan
            7. Giữ câu trả lời đầy đủ và chi tiết
            """.strip()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_functions_agent(llm=llm, tools=[tool], prompt=prompt)
            return AgentExecutor(
                agent=agent, 
                tools=[tool], 
                verbose=True,
                max_iterations=3,  # Limit iterations
                early_stopping_method="generate"  # Enable early stopping
            )
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            raise

def get_retriever(collection_name: str = "data_test") -> CustomRetriever:
    """Create retriever instance"""
    return CustomRetriever(collection_name)

def get_llm_and_agent(retriever) -> AgentExecutor:
    """Create agent manager instance"""
    agent_manager = AgentManager()
    return agent_manager.agent_executor

# Initialize agent if token exists
if os.getenv("GITHUB_TOKEN"):
    try:
        retriever = get_retriever()
        agent_executor = get_llm_and_agent(retriever)
        logger.info("Agent system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        agent_executor = None
else:
    logger.warning("GITHUB_TOKEN not found in environment variables")
    agent_executor = None
