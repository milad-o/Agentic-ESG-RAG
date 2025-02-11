# Agentic_RAG_App/main.py
# This is the main file for the Agentic_RAG_App
# It contains the FastAPI app, the main function, and the index function

# Imports
import os

from dotenv import load_dotenv

load_dotenv(override=True)

APP_API_KEY = os.getenv("APP_API_KEY")

import base64
import tempfile

WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
WX_URL = os.getenv("WX_URL")
OPOINT_TOKEN = os.getenv("OPOINT_TOKEN")
ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_CERT = os.getenv("ES_CERT")

# Certificate for Elasticsearch ----------------
## Decode the Base64 certificate
es_cert = base64.b64decode(ES_CERT).decode("utf-8")

## Write the decoded certificate to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp_cert_file:
    temp_cert_file.write(es_cert.encode("utf-8"))
    temp_cert_path = temp_cert_file.name

# FastAPI security setup ----------------------------
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def check_api_key(api_key: str = Security(api_key_header)):
    if api_key == APP_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


# FastAPI app setup --------------------------------
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

# LLM and Embedding model --------------------------
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url=WX_URL,
    apikey=IBM_CLOUD_API_KEY,
    project_id=WX_PROJECT_ID,
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.MAX_NEW_TOKENS: 250,
        GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
    },
)

embed_model = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    apikey=IBM_CLOUD_API_KEY,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=256, chunk_overlap=64
)

# Function/tools for the agent -------------------
import aiohttp
from typing import Dict, List, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.tools import tool


## Function to fetch articles from OPoint
@tool
async def search_news(
    main_keyword: str,
    additional_keywords: Optional[str] = None,
    number_of_articles: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetches recent news articles related to a specified company and additional keywords.

    Parameters:
    ----------
    main_keyword : str
    - The main keyword to search for in the articles. Usually the name of the company.

    additional_keywords : Optional[str], optional
    - Additional keywords to filter articles based on their text content. Default is None.

    number_of_articles : int, optional
    - The number of articles to fetch. Defaults to 5.

    Returns:
    -------
    List[Dict[str, Any]]
    """

    async def fetch_articles(search_term: str) -> List[Document]:
        """
        Helper function to send a search request to the Opoint API and process the results.

        Parameters:
        ----------
        search_term : str
            The search query to be sent to the Opoint API.

        Returns:
        -------
        List[Document]
            A list of Document objects representing the articles found.
        """
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
                async with session.post(
                    api_url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()

                results = response_data["searchresult"]["document"]

                docs = []

                for doc in results:
                    docs.append(
                        Document(
                            page_content=f"{doc['body']['text']}",
                            metadata={
                                "url": doc["orig_url"],
                                "rank_global": doc["site_rank"]["rank_global"],
                                "rank_country": doc["site_rank"]["rank_country"],
                            },
                        )
                    )
                return docs
            except aiohttp.ClientError as e:
                print(f"Error fetching articles: {e}")
                return None

    # Base Opoint API URL and authorization token
    api_url = "https://api.opoint.com/search/"
    opoint_token = OPOINT_TOKEN  # Replace with your Opoint token

    headers = {
        "Authorization": f"Token {opoint_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Initial search using header and body keywords
    search_term = f"header:{main_keyword}"
    if additional_keywords:
        search_term += f" body:{additional_keywords}"
    search_term += " profile:523024"
    search_term = f"{search_term}"

    documents = await fetch_articles(search_term)
    if documents:
        # Process and split documents if found
        doc_splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embed_model,
            collection_name="opoint",
        )
        return await vectorstore.asimilarity_search(search_term, k=3)

    # Notify the agent if no results are found in both searches
    return "No search results found."


## Elasticsearch function
from elasticsearch import AsyncElasticsearch

es_client = AsyncElasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs=temp_cert_path,
    verify_certs=True,
)

from typing import Dict, List, Any


@tool
async def search_documents(
    query_text: str, index: str = "index_m1", top_hits: int = 10
) -> List[Dict[str, Any]]:
    """
    Search the corporate documents database and returns a list of Documents with metadata.

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
        documents = [
            Document(
                page_content=hit["_source"]["body_content_field"],
                metadata={
                    "page_label": hit["_source"]["metadata"]["page_label"],
                    "file_name": hit["_source"]["file_name"],
                },
            )
            for hit in hits
        ]

        if not documents:
            print("No documents found in database.")
            return []

        doc_splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embed_model,
            collection_name="elasticsearch",
        )

        return await vectorstore.asimilarity_search(query_text, k=3)

    except Exception as e:
        print(f"Error searching Elasticsearch: {e}")
        return []


# Set up list of tools
tools = [search_documents, search_news]

# Agent prompt -----------------------------------------------
system_prompt = """Respond to the human as helpfully and accurately as possible.

You have access to the following tools: {tools}

You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask to achieve the goal.

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:"
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation

"""


human_prompt = """{input}
{agent_scratchpad}
(reminder to always respond in a JSON blob)"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human_prompt),
    ]
)

from langchain.tools.render import render_text_description_and_args

prompt = prompt.partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
)

# Agent memory -----------------------------------------------
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True  # Required for chat-style memory
)

base_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"])
    )
    | prompt
    | llm
    | JSONAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=base_chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    return_intermediate_steps=True,
)

agent_with_history = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=lambda session_id: memory.chat_memory,
    input_messages_key="input",  # Key for user input
    output_messages_key="output",  # Key for agent response
    history_messages_key="chat_history",  # Matches memory_key
)


# FastApi app setup ------------------------------------------------
@app.get("/")
def index():
    return {"Message": "Agentic RAG App is running"}


from pydantic import BaseModel, ValidationError


# Input model
class QueryRequest(BaseModel):
    question: str


## Agent Query Endpoint
@app.post("/query")
async def query_agent(request: QueryRequest, api_key: str = Security(check_api_key)):
    try:
        # Call the agent executor with the user input
        result = await agent_with_history.ainvoke(
            {"input": request.question},
            config={"configurable": {"session_id": "123456789"}},
        )
        return result

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


# Run the app
if __name__ == "__main__":
    import sys
    import uvicorn

    if "uvicorn" not in sys.argv[0]:
        uvicorn.run("main:app", host="0.0.0.0", port=4050, reload=True)
