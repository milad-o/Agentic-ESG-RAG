from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import os
from llama_index.llms.ibm import WatsonxLLM
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec

from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware

from elasticsearch import Elasticsearch
import json
import base64
import tempfile

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load environment variables
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
WX_URL = os.getenv("WX_URL")
ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
OPOINT_TOKEN = os.getenv("OPOINT_TOKEN")
ES_CERT = os.getenv("ES_CERT")


# Initialize LLM parameters
temperature = 0.1
max_new_tokens = 200
additional_params = {
    "decoding_method": "sample",
    "min_new_tokens": 1,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1.1,
}

llm = WatsonxLLM(
    model_id="mistralai/mistral-large",
    apikey=IBM_CLOUD_API_KEY,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)

embed_model = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    apikey=IBM_CLOUD_API_KEY,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
)

vector_memory = VectorMemory.from_defaults(
    vector_store=None,
    embed_model=embed_model,
    retriever_kwargs={"similarity_top_k": 1},
)

chat_memory_buffer = ChatMemoryBuffer.from_defaults()

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer,
    secondary_memory_sources=[vector_memory],
)

# Decode the Base64 certificate
es_cert = base64.b64decode(ES_CERT).decode("utf-8")

# Write the decoded certificate to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp_cert_file:
    temp_cert_file.write(es_cert.encode("utf-8"))
    temp_cert_path = temp_cert_file.name

es_client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs=temp_cert_path,
    verify_certs=True
)

from pydantic import BaseModel

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
    
    response = es_client.search(index=index, body=search_query)

    return response['hits']


es_tool = FunctionTool.from_defaults(search_corporate_reports)

es_tool_ls = LoadAndSearchToolSpec.from_defaults(es_tool).to_tool_list()


def fetch_news_articles(company_name, additional_keywords=None, number_of_articles=5):
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
        search_term += f" AND body:'{additional_keywords}'"

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
        response_data = response.json()

        # Extracting articles and relevant fields
        articles = []
        for article in response_data.get("searchresult", {}).get("document", []):
            articles.append(
                {
                    "header": article.get("header", {}).get("text", ""),
                    "text": article.get("body", {}).get("text", ""),
                    "url": article.get("orig_url", ""),
                    "rank_global": article.get("site_rank", {}).get(
                        "rank_global", None
                    ),
                    "rank_country": article.get("site_rank", {}).get(
                        "rank_country", None
                    ),
                }
            )

        return articles  # Convert the list to a JSON string

    except requests.RequestException as e:
        print(f"An error occurred while fetching articles: {e}")
        return []


opoint_tool = FunctionTool.from_defaults(fetch_news_articles)

opoint_tool_ls = LoadAndSearchToolSpec.from_defaults(opoint_tool).to_tool_list()

agent = ReActAgent(
    llm=llm,
    tools=[
        es_tool_ls[0],
        es_tool_ls[1],
        opoint_tool_ls[0],
        opoint_tool_ls[1],
    ],
    verbose=True,
    memory=composable_memory,
    max_iterations=20
)

from llama_index.core import PromptTemplate

react_system_header_str = """
You are an advanced assistance agent specializing in providing corporate ESG-related insights \
    and answering a variety of other questions. You are equipped to deliver thorough responses,\
    including summaries, analyses, and comprehensive answers based on relevant sources.\
    Your responses should always include references to the sources for any statements you make,\
    especially when data or recent information is involved.

## Tools

You have access to a diverse set of tools to help answer user questions effectively.\
    These include knowledge-base queries, news searches, and additional tools that\
    may be introduced later. It is your responsibility to decide the best approach\
    to answer each question, which may require breaking the task into subtasks and\
    using different tools in sequence.

You have access to the following tools:
{tool_desc}

## Output Format

To answer the question at hand, use the following format:
```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.
"""

system_prompt = PromptTemplate(react_system_header_str)

agent.update_prompts({"agent_worker:system_prompt": system_prompt})


@app.get("/")
def index():
    return {"Hello": "World"}

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
