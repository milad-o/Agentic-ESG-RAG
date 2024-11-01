from fastapi import FastAPI, HTTPException, Security, Depends
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
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware


# FastAPI app setup
app = FastAPI()


# Set up CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# Load environment variables
WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
WX_URL = os.getenv("WX_URL")

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


# Query knowledge base function
def query_knowledge_base(question: str) -> str:
    """
    Function to query the knowledge base and return the LLM response.
    """
    url = "https://rag-llm-llama-service-rel8ed.1mitp0uqijpc.us-south.codeengine.appdomain.cloud/queryLLM"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "RAG-APP-API-Key": "Rel8edRAGLLM",
    }

    payload = {
        "question": question,
        "es_index_name": "index_m1",
        "filters": {},
        "llm_params": {"inputs": []},
        "es_model_name": ".elser_model_2_linux-x86_64",
        "llm_instructions": "[INST]<<SYS>>You are a helpful AI assistant specialized in analyzing and presenting corporate ESG information and data. Answer the following question based on the documents provided. Respond concisely and ensure your response is accurate and clear. If the documents do not contain the answer, state that no relevant information is available.<</SYS>>\\n\\n{context_str}\\n\\n{query_str}[/INST]",
        "es_index_text_field": "body_content_field",
        "es_model_text_field": "ml.tokens",
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json().get("llm_response")
    except requests.exceptions.ConnectionError:
        return "Error: Failed to connect to the knowledge base. Please try again later."
    except requests.exceptions.HTTPError as err:
        return f"Error: {err.response.status_code}, {err.response.text}"


from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


es_query_tool = FunctionTool.from_defaults(query_knowledge_base)

OPOINT_TOKEN = os.getenv("OPOINT_TOKEN")


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


from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec

opoint_tool = FunctionTool.from_defaults(fetch_news_articles)

opoint_tool_ls = LoadAndSearchToolSpec.from_defaults(opoint_tool).to_tool_list()

agent = ReActAgent(
    llm=llm,
    tools=[es_query_tool, opoint_tool_ls[0], opoint_tool_ls[1]],
    verbose=True,
    memory=composable_memory,
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


@app.post("/query")
async def query_endpoint(request: QuestionRequest):
    try:
        # Query the agent and get the response as a string
        result = agent.chat(request.question)
        # Return the structured response and the processed raw_output
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
