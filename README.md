# Build-An-LLM-RAG-Chatbot-With-LangChain-Python

- Step 0: Install the required libraries (note, I may have listed missing libraries, so when running the code, if any libraries are missing, you can install them):
    + pip install langchain
    + pip install langchain-core
    + pip install langchain-community
    + pip install langchain-openai
    + pip install python-dotenv
    + pip install beautifulsoup4
    + pip install langchain_milvus
- Step 1: Run the command: docker compose up --build
    + Optionally, you can run attu to view the data on the UI, using the following command:
      + docker run -p 8000:3000 -e MILVUS_URL={milvus server IP}:19530 zilliz/attu:v2.4
      + Run the following command to get the IP address: ipconfig getifaddr en0
    + Learn more: 'https://github.com/zilliztech/attu'
- Step 2: Create a .env file in the src folder, access https://platform.openai.com/api-keys to create an api key, then paste it into the .env file.
    + For example: OPENAI_API_KEY=sk-proj-mzBm******1UYRb7cTqagfpCG
- Step 3: Access the src folder, run the crawl.py file with the syntax:
    + python crawl.py
- Step 4: Access the src folder, run the seed_data.py file to seed data into Milvus with syntax:
    + python seed_data.py
- Step 4: Access the src folder, run the agent.py file with the syntax:
    + streamlit run agent.py
