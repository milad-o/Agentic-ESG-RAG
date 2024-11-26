# Agentic_RAG_App/main.py
# This is the main file for the Agentic_RAG_App
# It contains the FastAPI app, the main function, and the index function

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
WD_URL = os.getenv("WD_URL")


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
import aiohttp
from typing import Dict, List, Any, Optional
from llama_index.core import Document

async def fetch_news_articles(
    company_name: str, additional_keywords: Optional[str] = None, number_of_articles=20
) -> List[Dict[str, Any]]:
    """
    Fetches recent news articles related to a specified company and additional keywords
    and returns a list of dictionaries representing chunks of the articles with metadata.

    Args:
    - company_name (str): The name of the company to fetch articles for.
    - additional_keywords (str, optional): Additional keywords to filter the articles by.
    - number_of_articles (int, optional): The number of articles to fetch. Defaults to 20.

    Returns:
    - List[Dict[str, Any]]: A list of Document objects, where each represents a chunk of the articles.
    """

    # Base Opoint API URL and authorization token
    api_url = "https://api.opoint.com/search/"
    opoint_token = OPOINT_TOKEN  # Replace with your Opoint token

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

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                response_data = await response.json()

            documents = response_data.get("searchresult", {}).get("document", [])
            results = [
                Document(
                    text=f"Heading: {doc['header']}\n\n{doc['body']['text']}",
                    metadata={
                        "url": doc.get("orig_url", ""),
                        "rank_global": doc.get("site_rank", {}).get("rank_global", None),
                        "rank_country": doc.get("site_rank", {}).get(
                            "rank_country", None
                        ),
                    },
                )
                for doc in documents
            ]
            return results

        except aiohttp.ClientError as e:
            print(f"Error fetching articles: {e}")
            return []


## Elasticsearch function
from elasticsearch import AsyncElasticsearch


## Elasticsearch function
es_client = AsyncElasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs=temp_cert_path,
    verify_certs=True,
)


from typing import Dict, List, Any


async def search_corporate_reports(
    query_text: str, index: str = "index_m1", top_hits: int = 10
) -> List[Dict[str, Any]]:
    """
    Search the corporate documents database and returns a list of dictionaries
    representing matching documents with metadata.

    Args:
    - query_text (str): The text to search for in the documents.
    - index (str, optional): The name of the Elasticsearch index to search. Defaults to "index_m1".
    - top_hits (int, optional): The maximum number of top hits to return. Defaults to 10.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries representing matching documents with metadata.
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

    try:
        response = await es_client.search(index=index, body=search_query)
        hits = response["hits"]["hits"]
        results = [
            Document(
                text=hit["_source"]["body_content_field"],
                metadata={
                    "page_label": hit["_source"]["metadata"]["page_label"],
                    "file_name": hit["_source"]["file_name"],
                },
            )
            for hit in hits
        ]
        return results

    except Exception as e:
        print(f"Error searching Elasticsearch: {e}")
        return []

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

# Benchmarking tools ----------------------------
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool

def get_company_metrics():
    """
    Returns a table of company metrics and industry codes.
    
    No arguments are required.

    """

    table = """
    companyName,industry_code,co2e_emission,MATERIALITY_INSIGHT,SUPPLY_CHAIN_INSIGHT,GHG_EMISSIONS_INSIGHT,LABOR_PRACTICES_INSIGHT,BUSINESS_ETHICS_INSIGHT,ENERGY_INSIGHT,DATA_SECURITY_INSIGHT
    Bruce Power,221,430901,68,65,74,57,38,69,31
    Ontario Power Generation,221,430901,57,62,67,42,35,75,34
    Northwest Natural Gas Co,221,187942,69,0,68,0,0,76,0
    "Burns & Mcdonnell, Inc.",221,187942,62,67,76,51,49,80,0
    Invista,325,47279,60,67,67,56,32,67,35
    """
    return table

def get_industry_averages():
    """
    Returns a table of industry averages.
    
    No arguments are required.

    """


    table = """
    Industry,Industry_code,co2 emission (sector),MATERIALITY_INSIGHT,SUPPLY_CHAIN_INSIGHT,GHG_EMISSIONS_INSIGHT,LABOR_PRACTICES_INSIGHT,BUSINESS_ETHICS_INSIGHT,ENERGY_INSIGHT,DATA_SECURITY_INSIGHT
    Mining and Quarrying (except Oil and Gas),212,"97,021",69,63,61,50,35,76,29
    Utilities,221,"430,901",57,62,67,42,35,75,34
    Paper Manufacturing,322,"187,942",62,67,76,51,49,80,
    Chemical Manufacturing,325,"142,012",59,67,66,51,41,66,38
    Fabricated Metal Product Manufacturing,332,"103,164",61,59,71,53,37,69,60
    Waste Management and Remediation Services,562,"47,279",60,67,67,56,32,67,35
    """

    return table
    
b_company_metrics_tool = FunctionTool.from_defaults(get_company_metrics)
b_industry_averages_tool = FunctionTool.from_defaults(get_industry_averages)


benchmark_qe = ReActAgent.from_tools(
    tools=[b_company_metrics_tool, b_industry_averages_tool],
    verbose=True,
)

benchmark_qe_tool = QueryEngineTool(
    query_engine=benchmark_qe,
    metadata=ToolMetadata(
        name="benchmark_qe_tool",
        description=
        """
        Use this agent to answer any questions regarding comparison
        between companies to their industry average.

        It can also be used to return industry averages.
        """,
    ),
)


# Memory for the agent ------------------------
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
from llama_index.core.agent import ReActAgent

agent = ReActAgent(
    llm=llm,
    tools=[
        *es_tool_ls, 
        *opoint_tool_ls, 
        benchmark_qe_tool
    ],
    verbose=True,
    memory=composable_memory,
    max_iterations=20,
    context="RULES: Use 'CoRel8ed average' for industry averages.",
)

# FastApi app setup ----------------------------
@app.get("/")
def index():
    return {"Message": "Agentic RAG API is running"}


## Main endpoint for querying the agent
from pydantic import BaseModel

@app.get("/")
def index():
    return {"Message": "Agentic RAG API is running"}

class QuestionRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(request: QuestionRequest):
    try:
        reply = await agent.achat(request.question)
        return {
            "response": reply.response,
            "sources": [item.dict() for item in reply.sources],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI with uvicorn ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4050, reload=True)
