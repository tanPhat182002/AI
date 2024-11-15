# crawl.py
import os
import re
import json
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentExtractor:
    """Enhanced content extraction with cleaning and validation"""
    @staticmethod
    def extract(html: str) -> str:
        """Extract and clean content from HTML"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for tag in ['script', 'style', 'nav', 'footer']:
                for element in soup.find_all(tag):
                    element.decompose()
                    
            # Extract text
            text = soup.get_text()
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            return ""

# Maintain compatibility with old code while using new implementation
def crawl_web(url: str) -> List[Document]:
    """Main crawl function for backward compatibility"""
    logger.info(f"Starting crawl from: {url}")
    try:
        # Use WebBaseLoader for simple URLs
        if 'viblo.asia' in url:
            loader = WebBaseLoader(url)
        else:
            # Use RecursiveUrlLoader for documentation sites
            loader = RecursiveUrlLoader(
                url=url,
                extractor=ContentExtractor.extract,
                max_depth=4,
                prevent_outside=True
            )
        
        docs = loader.load()
        logger.info(f'Retrieved {len(docs)} documents')
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500
        )
        all_splits = text_splitter.split_documents(docs)
        logger.info(f'Created {len(all_splits)} chunks')
        
        return all_splits
        
    except Exception as e:
        logger.error(f"Crawl failed: {str(e)}")
        return []

def web_base_loader(url_data: str) -> List[Document]:
    """Load from single URL"""
    try:
        loader = WebBaseLoader(url_data)
        docs = loader.load()
        logger.info(f'Retrieved {len(docs)} documents')
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500
        )
        all_splits = text_splitter.split_documents(docs)
        logger.info(f'Created {len(all_splits)} chunks')
        
        return all_splits
        
    except Exception as e:
        logger.error(f"Loading failed: {str(e)}")
        return []

def save_data_locally(documents: List[Document], filename: str, directory: str):
    """Save documents to JSON file"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        file_path = os.path.join(directory, filename)
        
        data_to_save = [{
            'page_content': doc.page_content,
            'metadata': doc.metadata
        } for doc in documents]
        
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data_to_save, file, indent=2, ensure_ascii=False)
            
        logger.info(f'Data saved to {file_path}')
        
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        documents = crawl_web('https://marc.com.vn/')
        if documents:
            save_data_locally(documents, 'stack.json', 'data')
            logger.info("Crawl and save completed successfully")
        else:
            logger.warning("No documents were crawled")
            
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")

if __name__ == "__main__":
    main()