import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from crawl import crawl_web

load_dotenv()

def load_data_from_local(filename, directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')


def seed_milvus(URI_link, collection_name, filename, directory):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    local_data, doc_name = load_data_from_local(filename, directory)

    documents = [
        Document(
            page_content=doc.get('page_content', ''),
            metadata={
                'source': doc['metadata'].get('source', ''),
                'content_type': doc['metadata'].get('content_type', ''),
                'title': doc['metadata'].get('title', ''),
                'description': doc['metadata'].get('description', ''),
                'language': doc['metadata'].get('language', '') or '',
                'doc_name': doc_name
            }
        )
        for doc in local_data
    ]

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True
    )
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore

def seed_milvus_live(URL, URI_link, collection_name, doc_name):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    data = crawl_web(URL)

    documents = [
        Document(
            page_content=doc.get('page_content', ''),
            metadata={
                'source': doc['metadata'].get('source', ''),
                'content_type': doc['metadata'].get('content_type', ''),
                'title': doc['metadata'].get('title', ''),
                'description': doc['metadata'].get('description', ''),
                'language': doc['metadata'].get('language', '') if doc['metadata'].get('language') else 'unknown',
                'doc_name': doc_name
            }
        )
        for doc in data
    ]

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        # drop_old=True
    )
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore

def connect_to_milvus(URI_link, collection_name):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def main():
    seed_milvus('http://localhost:19530', 'data_test', 'stack.json', 'data_v2')
    seed_milvus_live('abc', 'http://localhost:19530', 'data_test', '123')


if __name__ == "__main__":
    main()
