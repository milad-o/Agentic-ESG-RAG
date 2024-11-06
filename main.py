# Agentic_RAG_App/main.py
# This is the main file for the Agentic_RAG_App
# It contains the FastAPI app, the main function, and the index function

# If you're running this locally, use Uvicorn to serve the app

# FastAPI app setup ----------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables -------------------
import os
from dotenv import load_dotenv
load_dotenv()
import base64
import tempfile

WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
WX_URL = os.getenv("WX_URL")
OPOINT_TOKEN = "76bbbc556719f1d22b70dedfec56cd255d76b96e"  # the old one
ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_CERT = os.getenv("ES_CERT")

# Decode the Base64 certificate for Elasticsearch
es_cert = base64.b64decode(ES_CERT).decode("utf-8")

# Write the decoded certificate to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp_cert_file:
    temp_cert_file.write(es_cert.encode("utf-8"))
    temp_cert_path = temp_cert_file.name


# LLM and Embedding model ---------------------
from llama_index.llms.ibm import WatsonxLLM

llm = WatsonxLLM(
    model_id="mistralai/mistral-large",
    # model_id="meta-llama/llama-3-2-3b-instruct",
    apikey=IBM_CLOUD_API_KEY,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
    temperature=.1,
    additional_params={"top_p": 1},
    max_new_tokens=200
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

## Setting default llm and embedding model for llama_index
from llama_index.core import Settings
Settings.embed_model = embed_model
Settings.llm = llm


# Function/tools for the agent -------------------
## Opoint function
import requests
import json

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
node_parser = SentenceSplitter(chunk_size=150, chunk_overlap=20)

from chromadb import EphemeralClient
from llama_index.vector_stores.chroma import ChromaVectorStore

client = EphemeralClient()

collection_opoint = client.get_or_create_collection(name="opoint")
chroma_store_opoint = ChromaVectorStore(chroma_collection=collection_opoint)
storage_context_opoint = StorageContext.from_defaults(vector_store=chroma_store_opoint)

from typing import List, Dict, Any, Optional

def fetch_news_articles(
        company_name: str,
        additional_keywords: Optional[str]=None,
        number_of_articles: int=5
    ) -> List[Dict[str, Any]]:
    """
    Fetches recent news articles related to a specified company from the Opoint API.

    ### Parameters:
    - `company_name` (str): The primary company name to search for in news articles.
    - `additional_keywords` (str, optional): Extra keywords to narrow down the search for articles.
      Can be any term that refines the search context, like "distribution" or "sales".
    - `number_of_articles` (int, default=5): The number of news articles to return.

    ### Returns:
    - A list of dictionaries, each containing:
      - `header` (str): The headline of the news article.
      - `text` (str): The test body of the article.
      - `url` (str): Direct URL link to the article.
      - `rank_global` (int): Global rank of the source website.
      - `rank_country` (int): Country-specific rank of the source website.

    ### Usage Example:
    `fetch_news_articles("Giant Eagle", additional_keywords="sustainability", number_of_articles=3)`

    ### Agent Instructions:
    - Call this function by providing `company_name`, `additional_keywords` (optional), and `number_of_articles` (optional).
    - Use the `company_name` parameter to specify the main focus of the search (e.g., "Giant Eagle").
    - Use `additional_keywords` if narrowing down the search to specific themes or topics within articles is needed.
    - The function will return a structured list of articles with specific fields, which can be parsed or presented to end-users.
    - Fields `header`, `summary`, and `url` should be prioritized for article display, while `rank_global` and `rank_country` can provide context on the article's source.
    - Make sure, by default, you provide the references.

    ### Notes for the AI Agent:
    - Ensure that the API token is valid before calling the function.
    - If any error occurs in fetching data, log the error and inform the user.
    - Return an empty list if no articles are found, and log this as well.
    """

    # Base Opoint API URL and authorization token
    api_url = "https://api.opoint.com/search/"
    opoint_token = OPOINT_TOKEN  # replace with your Opoint token

    headers = {
        "Authorization": f"Token {opoint_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Constructing the search term based on company name and additional keywords
    search_term = f"header:'{company_name}'"
    if additional_keywords:
        search_term += f" AND body:{additional_keywords}"

    # Defining payload with query parameters
    payload = {
        "searchterm": search_term,
        "params": {
            "requestedarticles": number_of_articles,
            "main": {"header": 1, "summary": 1, "text": 1},
        },
        "profile": 523024,
    }

    try:
        # Sending the POST request to the API
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()['searchresult']['document']

        # Extracting articles and relevant fields
        articles = []
        for article in response_data:
            articles.append(
                {
                    "header": article.get("header", {}).get("text", ""),
                    "text": article.get("body", {}).get("text", ""),
                    "metadata": {
                        "url": article.get("orig_url", ""),
                        "rank_global": article.get("site_rank", {}).get(
                            "rank_global", None
                        ),
                        "rank_country": article.get("site_rank", {}).get(
                            "rank_country", None
                        ),
                    },
                }
            )

        docs = []

        for article in articles:
            doc = Document(
                text=f"HEADER: {article['header']}\n\nTEXT: {article['text']}",
                metadata = article['metadata'],
            )
            docs.append(doc)

        nodes = node_parser.get_nodes_from_documents(docs)

        index = VectorStoreIndex(nodes, storage_context=storage_context_opoint)

        retriever = index.as_retriever(similarity_top_k=5)

        output = retriever.retrieve(str_or_query_bundle=f"{company_name} {additional_keywords}")

        result = []
        for n in output:
            result.append(
            {
                "text": n.node.text,
                "metadata": n.node.metadata
            }
        )

        return result

    except requests.RequestException as e:
        print(f"An error occurred while fetching articles: {e}")
        return []


## Elasticsearch function
from elasticsearch import Elasticsearch

es_client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs=temp_cert_path,
    verify_certs=True
)

collection_elastic = client.get_or_create_collection(name="elastic")
chroma_store_elastic = ChromaVectorStore(chroma_collection=collection_elastic)

storage_context_elastic = StorageContext.from_defaults(vector_store=chroma_store_elastic)

def search_corporate_reports(query_text: str, index: str = "index_m1", top_hits: int = 5):
    """
    Search the corporate documents database on an Elasticsearch index using a semantic text expansion query.

    Parameters:
    - query_text (str): The input text for the semantic search, e.g., "Google sustainability".
    - index (str, optional): The name of the Elasticsearch index to query. Defaults to "index_m1".
    - top_hits (int, optional): The maximum number of top hits to return. Defaults to 5.

    Returns:
    - list of dict: A list of dictionaries, where each dictionary represents a matching document and includes:
      "metadata.page_label", "file_name", and "body_content_field".

    Example:
    --------
    response = search_corporate_docs_database("Google sustainability")
    for doc in response:
        print(doc)

    Notes:
    - This function is designed as a tool for a ReActAgent to perform semantic searches in a corporate document database.
    - Uses a fixed model_id: ".elser_model_2_linux-x86_64" for the text expansion query.
    - Only relevant document fields are returned for succinct responses.
    """
        
    search_query = {
        "query": {
            "text_expansion": {
                "ml.tokens": {
                    "model_id": ".elser_model_2_linux-x86_64",
                    "model_text": query_text
                }
            }
        },
        "_source": ["metadata.page_label", "file_name", "body_content_field"],
        "size": top_hits,
        "track_total_hits": False
    }
    
    hits = es_client.search(index=index, body=search_query)['hits']['hits']

    docs = []

    for hit in hits:
        doc = Document(
            text=hit['_source']['body_content_field'],
            metadata = {
                "page_label": hit['_source']['metadata']['page_label'],
                "file_name": hit['_source']['file_name']
            },
            doc_id=hit['_id']  
        )
        docs.append(doc)

    nodes = node_parser.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes, storage_context=storage_context_elastic)

    retriever = index.as_retriever(similarity_top_k=5)

    output = retriever.retrieve(str_or_query_bundle=query_text)

    result = []
    for n in output:
        result.append(
            {
                "text": n.node.text,
                "metadata": n.node.metadata
            }
        )

    return result

# Agent tools and subsequent L&S tools
from llama_index.core.tools import FunctionTool

es_tool = FunctionTool.from_defaults(search_corporate_reports)
opoint_tool = FunctionTool.from_defaults(fetch_news_articles)


# Memory for the agent ------------------------
from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)

vector_memory = VectorMemory.from_defaults(
    vector_store=None,
    embed_model=embed_model,
    retriever_kwargs={"similarity_top_k": 3},
)

chat_memory_buffer = ChatMemoryBuffer.from_defaults()

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer,
    secondary_memory_sources=[vector_memory],
)

# The agent ----------------------------------
from llama_index.core.agent import ReActAgent


context = """
RULES:
- Use the tools to answer user queries.
- Corporate reports should be used in preference to news articles.
- Use least number of tools.
- Don't use the same tool with the same parameters twice. 
- Don't use the tools if you already know the answer.
- Provide references.
"""

agent = ReActAgent(
    llm=llm,
    tools=[
        es_tool,
        opoint_tool
    ],
    verbose=True,
    memory=composable_memory,
    max_iterations=12,
    context=context
)


# FastApi app setup ----------------------------
@app.get("/")
def index():
    return {"Hello": "World"}

## Main endpoint for querying the agent
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(request: QuestionRequest):
    try:
        # Query the agent and get the response as a string
        reply = agent.chat(request.question)
        # Return the structured response and the processed raw_output
        result = {
            "response": reply.response,
            "sources": [item.dict() for item in reply.sources]
            }
        
        return result
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to the knowledge base. Please try again later.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# If you're running this locally, use Uvicorn to serve the app
if __name__ == "__main__":
    import sys
    import uvicorn

    if "uvicorn" not in sys.argv[0]:
        uvicorn.run("main:app", host="0.0.0.0", port=4050, reload=True)
