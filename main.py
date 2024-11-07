# Agentic_RAG_App/main.py
# This is the main file for the Agentic_RAG_App
# It contains the FastAPI app, the main function, and the index function

# If you're running this locally, use `uvicorn` to serve the app

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
    apikey=IBM_CLOUD_API_KEY,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
)

from llama_index.embeddings.ibm import WatsonxEmbeddings

embed_model = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    apikey=IBM_CLOUD_API_KEY,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
    truncate_input_tokens= 10
)

## Setting default llm and embedding model for llama_index
from llama_index.core import Settings

Settings.embed_model = embed_model
Settings.llm = llm


# Function/tools for the agent -------------------
## Opoint function
import requests

from typing import List, Dict, Any, Optional

def fetch_news_articles(
    company_name: str, additional_keywords: Optional[str] = None, number_of_articles=5
) -> List[Dict[str, Any]]:
    """
    Fetches recent news articles related to a specified company and additional keywords
    and returns a list of dictionaries representing chunks of the articles with metadata.

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
    search_term += " globalrank<1000 profile:523024"
    search_term = f"({search_term})"

    # Defining payload with query parameters
    payload = {
        "searchterm": search_term,
        "params": {
            "requestedarticles": number_of_articles,
            "main": {"header": 1, "text": 1},
            "groupidentical": True,
        },
    }

    try:
        # Sending the POST request to the API
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()["searchresult"]["document"]

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

        return articles

    except requests.RequestException as e:
        print(f"An error occurred while fetching articles: {e}")
        return []


## Elasticsearch function
from elasticsearch import Elasticsearch

es_client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs=temp_cert_path,
    verify_certs=True,
)


def search_corporate_reports(
    query_text: str, index: str = "index_m1", top_hits: int = 5
):
    """
    Search the corporate documents database and returns a list of dictionaries
    representing matching documents with metadata.
    """

    search_query = {
        "query": {
            "text_expansion": {
                "ml.tokens": {
                    "model_id": ".elser_model_2_linux-x86_64",
                    "model_text": query_text,
                }
            }
        },
        "_source": ["metadata.page_label", "file_name", "body_content_field"],
        "size": top_hits,
        "track_total_hits": False,
    }

    hits = es_client.search(index=index, body=search_query)["hits"]["hits"]

    return hits


# Agent tools and subsequent L&S tools
from llama_index.core.tools import FunctionTool

es_tool = FunctionTool.from_defaults(search_corporate_reports)
opoint_tool = FunctionTool.from_defaults(fetch_news_articles)

from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec

es_tool_ls = LoadAndSearchToolSpec.from_defaults(es_tool).to_tool_list()
opoint_tool_ls = LoadAndSearchToolSpec.from_defaults(opoint_tool).to_tool_list()


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

agent = ReActAgent(
    llm=llm,
    tools=[*es_tool_ls, *opoint_tool_ls],
    verbose=True,
    memory=composable_memory,
    max_iterations=12,
)


# FastApi app setup ----------------------------
@app.get("/")
def index():
    return {"Message": "Agentic RAG API is running"}


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
            "sources": [item.dict() for item in reply.sources],
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
