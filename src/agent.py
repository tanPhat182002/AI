from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from seed_data import seed_milvus, connect_to_milvus
import os
from dotenv import load_dotenv

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()


def get_retriever():
    vectorstore = connect_to_milvus('http://localhost:19530', 'data_test')
    milvus_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    documents = [Document(page_content=doc.page_content, metadata=doc.metadata)
                 for doc in vectorstore.similarity_search("", k=100)]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[milvus_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever


tool = create_retriever_tool(
    get_retriever(),
    "find",
    """
    Search for information of Stack AI.
    """
)


def get_llm_and_agent(_retriever):
    llm = ChatOpenAI(temperature=0,
                     streaming=True, model="gpt-4o")
    tools = [tool]
    system = """You are an expert at AI. Your name is ChatchatAI."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)

msgs = StreamlitChatMessageHistory(key="langchain_messages")

st.title("💬 AI Assistant")
st.caption("🚀 A Streamlit chatbot powered by LangChain and OpenAI")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    msgs.add_ai_message("How can I help you?")

for msg in st.session_state.messages:
    role = "assistant" if msg["role"] == "assistant" else "human"
    st.chat_message(role).write(msg["content"])

if prompt := st.chat_input("Ask me anything about Stack AI and related topics!"):
    st.session_state.messages.append({"role": "human", "content": prompt})
    st.chat_message("human").write(prompt)
    msgs.add_user_message(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())

        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]
        ]

        response = agent_executor.invoke(
            {
                "input": prompt,
                "chat_history": chat_history
            },
            {"callbacks": [st_callback]}
        )

        output = response["output"]
        st.session_state.messages.append({"role": "assistant", "content": output})
        msgs.add_ai_message(output)
        st.write(output)
