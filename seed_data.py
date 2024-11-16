# seed_data.py
import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache
from uuid import uuid4

from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from crawl import crawl_web

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubEmbeddings:
    """GitHub AI compatible embeddings"""
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is not set")

        self.api_key = token
        self.model = "text-embedding-3-large"
        self.api_base = "https://models.inference.ai.azure.com"
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            embedding_ctx_length=8191,
            chunk_size=1000,
            max_retries=3
        )
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
            
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Query embedding error: {str(e)}")
            raise
            
    def dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "model": self.model,
            "api_base": self.api_base,
            "embedding_ctx_length": 8191,
            "chunk_size": 1000,
            "max_retries": 3
        }

class MilvusManager:
    """Milvus database manager"""
    def __init__(self, uri: str = "http://localhost:19530"):
        self.uri = uri
        self.dimension = 3072
        self.max_retries = 3
        self.retry_delay = 1
        
    def create_collection(self, collection_name: str) -> Collection:
        """Create Milvus collection with optimized settings"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=500)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Document store",
            enable_dynamic_field=True
        )

        try:
            logger.info(f"Creating collection: {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # Optimized index parameters
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_SQ8",
                "params": {
                    "nlist": 1024,
                    "nprobe": 16
                }
            }
            
            logger.info("Creating index...")
            collection.create_index(field_name="vector", index_params=index_params)
            collection.load()
            logger.info("Collection created and loaded successfully")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    def reset_collection(self, collection_name: str) -> Collection:
        """Reset and recreate collection"""
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                timeout=30,
                pool_size=10
            )
            
            if utility.has_collection(collection_name):
                logger.info(f"Dropping existing collection: {collection_name}")
                utility.drop_collection(collection_name)
                time.sleep(2)  # Wait for cleanup
                
            logger.info(f"Creating new collection: {collection_name}")
            return self.create_collection(collection_name)
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise
        finally:
            try:
                connections.disconnect("default")
            except Exception as e:
                logger.warning(f"Error disconnecting: {str(e)}")
class DocumentProcessor:
    """Document processing and chunking"""
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        processed_docs = []
        
        for idx, doc in enumerate(documents):
            try:
                content = self._clean_content(doc.page_content)
                if not content:
                    logger.warning(f"Empty content in document {idx}")
                    continue
                    
                chunks = self.text_splitter.split_text(content)
                logger.info(f"Document {idx} split into {len(chunks)} chunks")
                
                processed_docs.extend([
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_size': len(chunk),
                            'chunk_id': chunk_idx,  # Thay chunk_index bằng chunk_id
                            'processing_timestamp': time.time(),
                            'original_doc_index': idx
                        }
                    )
                    for chunk_idx, chunk in enumerate(chunks)
                ])
                
            except Exception as e:
                logger.warning(f"Failed to process document {idx}: {str(e)}")
                continue
        return processed_docs
    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content"""
        if not content or not content.strip():
            return ""
        
        # Basic cleaning
        content = content.strip()
        content = ' '.join(content.split())
        content = content.replace('\t', ' ')
        
        # Remove control characters except newlines
        content = ''.join(char for char in content if ord(char) >= 32 or char == '\n')
        
        return content.strip()

class VectorStoreManager:
    """Manager for vector store operations"""
    def __init__(self, uri: str, collection_name: str, embeddings):
        self.uri = uri
        self.collection_name = collection_name
        self.batch_size = 10
        self.embeddings = embeddings
        self.vectorstore = self._create_vectorstore()
        
    def _create_vectorstore(self) -> Milvus:
        """Create Milvus vectorstore"""
        try:
            return Milvus(
                embedding_function=self.embeddings,
                connection_args={
                    "uri": self.uri,
                    "pool_size": 10,
                    "timeout": 30
                },
                collection_name=self.collection_name,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                enable_dynamic_field=True
            )
        except Exception as e:
            logger.error(f"Failed to create vectorstore: {str(e)}")
            raise
        
    def add_documents(self, documents: List[Document]):
        """Add documents in batches with retry logic"""
        total_batches = (len(documents) - 1) // self.batch_size + 1
        successful_batches = 0
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            try:
                texts = [doc.page_content for doc in batch]
                # Chuyển chunk_index thành chunk_id trong metadata
                metadatas = [{
                    **doc.metadata,
                    'chunk_id': doc.metadata.get('chunk_index', 0)  # Sử dụng chunk_index làm chunk_id
                } for doc in batch]
                ids = [str(uuid4()) for _ in range(len(batch))]
                
                self._add_texts_with_retry(texts, metadatas, ids)
                successful_batches += 1
                logger.info(f'Processed batch {i//self.batch_size + 1}/{total_batches}')
                
            except Exception as e:
                logger.error(f'Error in batch {i//self.batch_size + 1}: {str(e)}')
                continue
                
        logger.info(f"Successfully processed {successful_batches}/{total_batches} batches")
                
    def _add_texts_with_retry(self, texts: List[str], metadatas: List[Dict], ids: List[str], max_retries: int = 3):
        """Add texts with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Retry attempt {attempt + 1} after {wait_time}s delay")
                time.sleep(wait_time)

class CustomRetriever:
    """Custom retriever for document search"""
    def __init__(self, collection_name: str = "data_test"):
        self.collection_name = collection_name
        self.default_doc = self._create_default_doc()
        self._embeddings = None
        self._vectorstore = None
        
    def _create_default_doc(self) -> Document:
        """Create default response document"""
        return Document(
            page_content="""
            Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về Stack AI và các chủ đề liên quan đến AI, Machine Learning. 
            Đối với các câu hỏi khác, bạn nên tham khảo các trang web chuyên biệt.
            """.strip(),
            metadata={
                "source": "default",
                "type": "system_message",
                "timestamp": time.time()
            }
        )
        
    def _initialize_components(self):
        """Initialize embeddings and vectorstore"""
        try:
            if not self._embeddings:
                self._embeddings = GitHubEmbeddings()
                
            if not self._vectorstore:
                self._vectorstore = connect_to_milvus(
                    'http://localhost:19530',
                    self.collection_name,
                    self._embeddings
                )
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents"""
        try:
            self._initialize_components()
            results = self._vectorstore.similarity_search(query, k=4)
            
            if not results:
                logger.warning(f"No matching documents found for query: {query}")
                return [self.default_doc]
                
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [self.default_doc]

def create_embeddings(use_ollama: bool = False):
    """Factory function for creating embeddings"""
    try:
        if use_ollama:
            return OllamaEmbeddings(model="llama2")
        return GitHubEmbeddings()
    except Exception as e:
        logger.error(f"Failed to create embeddings: {str(e)}")
        raise

@lru_cache()
def connect_to_milvus(uri: str, collection_name: str, embeddings=None) -> Milvus:
    """Connect to Milvus with connection pooling and caching"""
    logger.info(f"Connecting to Milvus collection: {collection_name}")
    try:
        connections.connect(
            alias="default",
            uri=uri,
            timeout=30,
            pool_size=10
        )
        
        if not utility.has_collection(collection_name):
            logger.info(f"Creating new collection: {collection_name}")
            milvus_manager = MilvusManager(uri)
            milvus_manager.create_collection(collection_name)
            
        if embeddings is None:
            embeddings = GitHubEmbeddings()
            
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={
                "uri": uri,
                "pool_size": 10,
                "timeout": 30
            },
            collection_name=collection_name,
            text_field="content",
            vector_field="vector",
            primary_field="id",
            enable_dynamic_field=True
        )
        
        logger.info("Successfully connected to Milvus")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Milvus connection error: {str(e)}")
        raise
    finally:
        try:
            connections.disconnect("default")
        except Exception as e:
            logger.warning(f"Error disconnecting: {str(e)}")

def load_documents_from_file(file_path: str) -> List[Document]:
    """Load and validate documents from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            
        if not raw_data:
            raise ValueError(f"Empty data in file: {file_path}")
            
        return [
            Document(
                page_content=doc.get('page_content', '').strip(),
                metadata={
                    'source': doc.get('metadata', {}).get('source', ''),
                    'title': doc.get('metadata', {}).get('title', ''),
                    'doc_name': os.path.splitext(os.path.basename(file_path))[0],
                    'load_timestamp': time.time()
                }
            )
            for doc in raw_data
            if doc.get('page_content', '').strip()
        ]
        
    except Exception as e:
        logger.error(f"File loading error - {file_path}: {str(e)}")
        raise

def seed_milvus(uri: str, collection_name: str, filename: str, directory: str, use_ollama: bool = False) -> Milvus:
    """Seed Milvus from local file"""
    logger.info(f"Starting seed process for collection: {collection_name}")
    try:
        # Initialize components
        milvus_manager = MilvusManager(uri)
        doc_processor = DocumentProcessor()
        
        # Reset collection
        milvus_manager.reset_collection(collection_name)
        
        # Initialize embeddings
        embeddings = create_embeddings(use_ollama)
        
        # Load and process documents
        file_path = os.path.join(directory, filename)
        documents = load_documents_from_file(file_path)
        
        # Update metadata
        for doc in documents:
            doc.metadata.update({
                'file_path': file_path,
                'embedding_type': 'ollama' if use_ollama else 'github',
                'processing_timestamp': time.time()
            })
        
        # Process documents
        chunked_docs = doc_processor.process_documents(documents)
        
        if not chunked_docs:
            raise ValueError("No valid documents to process")
            
        logger.info(f'Original documents: {len(documents)}')
        logger.info(f'Processed chunks: {len(chunked_docs)}')
        
        # Initialize vector store and add documents
        vector_manager = VectorStoreManager(uri, collection_name, embeddings)
        vector_manager.add_documents(chunked_docs)
        
        logger.info("Seed process completed successfully")
        return vector_manager.vectorstore
        
    except Exception as e:
        logger.error(f"Seed process failed: {str(e)}")
        raise

def seed_milvus_live(url: str, uri: str, collection_name: str, doc_name: str, use_ollama: bool = False) -> Milvus:
    """Seed Milvus from live URL"""
    logger.info(f"Starting live seed process for URL: {url}")
    try:
        # Initialize components
        milvus_manager = MilvusManager(uri)
        doc_processor = DocumentProcessor()
        
        # Reset collection
        milvus_manager.reset_collection(collection_name)
        
        # Initialize embeddings
        embeddings = create_embeddings(use_ollama)
        
        # Crawl documents
        documents = crawl_web(url)
        if not documents:
            raise ValueError(f"No documents retrieved from URL: {url}")
            
        # Update metadata
        for doc in documents:
            doc.metadata.update({
                'source': doc.metadata.get('source', ''),
                'title': doc.metadata.get('title', ''),
                'doc_name': doc_name,
                'crawl_timestamp': time.time(),
                'url': url,
                'embedding_type': 'ollama' if use_ollama else 'github'
            })
        
        # Process documents
        chunked_docs = doc_processor.process_documents(documents)
        
        logger.info(f'Original documents: {len(documents)}')
        logger.info(f'Processed chunks: {len(chunked_docs)}')
        
        # Initialize vector store and add documents
        vector_manager = VectorStoreManager(uri, collection_name, embeddings)
        vector_manager.add_documents(chunked_docs)
        
        logger.info("Live seed process completed successfully")
        return vector_manager.vectorstore
        
    except Exception as e:
        logger.error(f"Live seed process failed: {str(e)}")
        raise

def get_retriever(collection_name: str = "data_test") -> CustomRetriever:
    """Create and initialize retriever"""
    try:
        retriever = CustomRetriever(collection_name)
        retriever._initialize_components()
        return retriever
    except Exception as e:
        logger.error(f"Failed to create retriever: {str(e)}")
        raise

def verify_collection(uri: str, collection_name: str) -> bool:
    """Verify collection existence and configuration"""
    logger.info(f"Verifying collection: {collection_name}")
    try:
        connections.connect(
            alias="default",
            uri=uri,
            timeout=30,
            pool_size=10
        )
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            schema = collection.schema
            
            logger.info(f"Collection '{collection_name}' configuration:")
            logger.info(f"Fields: {[field.name for field in schema.fields]}")
            logger.info(f"Vector dimension: {schema.fields[1].params['dim']}")
            
            required_fields = {"id", "vector", "content", "chunk_id", "source"}
            existing_fields = {field.name for field in schema.fields}
            
            if not required_fields.issubset(existing_fields):
                missing_fields = required_fields - existing_fields
                logger.warning(f"Missing required fields: {missing_fields}")
                return False
            
            return True
            
        logger.warning(f"Collection '{collection_name}' does not exist")
        return False
        
    except Exception as e:
        logger.error(f"Collection verification failed: {str(e)}")
        return False
    finally:
        try:
            connections.disconnect("default")
        except Exception as e:
            logger.warning(f"Error disconnecting: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("=== Starting Milvus System Test ===")
        
        URI = 'http://localhost:19530'
        TEST_COLLECTION = 'data_test'
        
        if verify_collection(URI, TEST_COLLECTION):
            logger.info("System check passed")
            
            try:
                vectorstore = seed_milvus(
                    URI,
                    TEST_COLLECTION,
                    'stack.json',
                    'data'
                )
                logger.info("Local file import successful")
                
            except Exception as e:
                logger.error(f"Local file import failed: {str(e)}")
            
            try:
                TEST_URL = 'https://www.stack-ai.com/docs'
                vectorstore_live = seed_milvus_live(
                    TEST_URL,
                    URI,
                    'data_test_live',
                    'stack-ai'
                )
                logger.info("Live URL import successful")
                
            except Exception as e:
                logger.error(f"Live URL import failed: {str(e)}")
            
        else:
            logger.error("System check failed")
            
        logger.info("=== Test sequence completed ===")
        
    except Exception as e:
        logger.error(f"Test sequence failed: {str(e)}")
        raise
