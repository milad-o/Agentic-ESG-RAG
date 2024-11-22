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

from llama_index.core import Document
from typing import List, Dict, Any, Optional

def fetch_news_articles(
    company_name: str, additional_keywords: Optional[str] = None, number_of_articles=20
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

        results = []
        for article in articles:
            results.append(
                Document(
                    text=f"Heading: article['text']\n\n{article['text']}",
                    metadata=article["metadata"],
                )
            )

        return results

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
    query_text: str, index: str = "index_m1", top_hits: int = 10
) -> List[Dict[str, Any]]:
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

    results = []

    for hit in hits:
        results.append(
            Document(
                text=hit["_source"]["body_content_field"],
                metadata={
                    "page_label": hit["_source"]["metadata"]["page_label"],
                    "file_name": hit["_source"]["file_name"],
                },
            )
        )

    return results


# Agent tools and subsequent L&S tools
from llama_index.core.tools import FunctionTool

es_tool = FunctionTool.from_defaults(search_corporate_reports)
opoint_tool = FunctionTool.from_defaults(fetch_news_articles)

from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec

es_tool_ls = LoadAndSearchToolSpec.from_defaults(es_tool).to_tool_list()
opoint_tool_ls = LoadAndSearchToolSpec.from_defaults(opoint_tool).to_tool_list()


# WD Sub-Agent functions and tools
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

WD_URL = os.getenv("WD_URL")
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")

authenticator = IAMAuthenticator(IBM_CLOUD_API_KEY)

discovery = DiscoveryV2(version="2023-03-31", authenticator=authenticator)

discovery.set_service_url(WD_URL)


def get_wd_projects() -> List[Dict[str, Any]]:
    """
    Use this function to fetch the list of projects from the Watson Discovery.
    """

    projects = discovery.list_projects().get_result()["projects"]
    return projects


def get_wd_collections(project_id: str) -> List[Dict[str, Any]]:
    """
    Use this function to fetch the list of collections for a specific project from the Watson Discovery.
    """

    collections = discovery.list_collections(project_id).get_result()["collections"]
    return collections


def get_wd_fields(project_id: str, collection_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Instructions
    ------------
    Use this function to fetch the list of fields for a specific project-collection from the Watson Discovery.
    """

    response = discovery.list_fields(
        project_id=project_id, collection_ids=collection_ids
    ).get_result()["fields"]

    fields = []

    for field in response:
        if field["field"].find("enriched_") == -1:
            fields.append({field["field"]: field["type"]})
    return fields


from typing import List


def get_wd_aggregated_results(
    variable: str,
    agg_func: str,
    project_id: str = "b2c1b89a-5841-446b-b3f7-e569d3170e32",
    collection_ids: List[str] = ["026b499f-57fc-edaf-0000-019291a70408"],
) -> dict:
    """
    Returns the aggregated results for a variable for a given project,
    collection, and filter from Watson Discovery.

    Direction for using this function:
    ----------
    Start with finding the correct `variable` from respected `get_wd_fields` function.

    Args
    ----------
    `variable`: list of all variables can be returned using the `get_wd_fields` function

    `project_id`: list of all project ids can be returned using the `get_project_id` function

    `collection_ids`: list of all collection ids can be returned using the `get_collection_id` function

    `agg_func`: the aggregation function can be: "average", "sum", "min", "max", "sum"

    Factset Insight
    ----------
    - project_id = 'b2c1b89a-5841-446b-b3f7-e569d3170e32'
    - collection_ids = ['026b499f-57fc-edaf-0000-019291a70408']
    """
    response = discovery.query(
        project_id=project_id,
        collection_ids=collection_ids,
        aggregation=f"{agg_func}({variable})",
    ).get_result()

    result = {
        "matching_results": response["matching_results"],
        "value": response["aggregations"][0]["value"],
    }

    return result


get_wd_projects_tool = FunctionTool.from_defaults(get_wd_projects)
get_wd_collections_tool = FunctionTool.from_defaults(get_wd_collections)
get_wd_fields_tool = FunctionTool.from_defaults(get_wd_fields)
get_wd_aggregated_results_tool = FunctionTool.from_defaults(get_wd_aggregated_results)

from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core import VectorStoreIndex

wd_tools = [
    get_wd_projects_tool,
    get_wd_collections_tool,
    get_wd_fields_tool,
    get_wd_aggregated_results_tool,
]

wd_obj_index = ObjectIndex.from_objects(
    wd_tools,
    node_mapping=SimpleToolNodeMapping.from_objects(wd_tools),
    index_cls=VectorStoreIndex,
)

wd_obj_retriever = wd_obj_index.as_retriever()

from llama_index.core.agent import ReActAgent

## Watson Discovery agent and agent-tool
wd_agent = ReActAgent.from_tools(
    tool_retriever=wd_obj_retriever, verbose=True, max_iterations=20,
    context="""
    Watson Discovery hierarchy: Projects -> Collections -> Documents -> Fields
    """
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata


wd_qe_tool = QueryEngineTool(
    query_engine=wd_agent,
    metadata=ToolMetadata(
        name="wd_qe_tool",
        description="""
        Use this agent to answer questions about Watson Discovery.

        It can also be used to return industry averages.
        """,
    ),
)


# Memory for the agent ------------------------
from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)

from chromadb import EphemeralClient
from llama_index.vector_stores.chroma import ChromaVectorStore

client = EphemeralClient()
memory_chroma_collection = client.get_or_create_collection("agent_memory")
memory_vector_store = ChromaVectorStore(memory_chroma_collection)

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

# Main Agent ----------------------------------

agent = ReActAgent(
    llm=llm,
    tools=[
        *es_tool_ls, 
        *opoint_tool_ls,
        # wd_qe_tool # Deactivated for now
    ],
    verbose=True,
    memory=composable_memory,
    max_iterations=20,
    context="""
    You are a top-level agent designed to choose the most appropriate 
    tool or agent to answer a user's question.
    """,
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
