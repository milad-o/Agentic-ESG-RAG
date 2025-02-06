# ESG Query Agent 🤖

A FastAPI-powered agentic AI chatbot specializing in corporate [ESG](https://en.wikipedia.org/wiki/Environmental,_social,_and_governance) (Environmental, Social, Governance) analysis, leveraging LangChain and IBM Watsonx.ai models.

[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge)](https://python.langchain.com/)
[![IBM Granite](https://img.shields.io/badge/IBM_Granite-052FAD?style=for-the-badge&logo=ibm)](https://www.ibm.com/granite/docs/models/granite/)

## Table of Contents
- [Features](#Features)
- [Deployment](#Deployment)
- [Configuration](#Configuration)
- [API Documentation](#API-Documentation)
- [Technology Stack](#Technology-Stack)
- [Acknowledgements](#Acknowledgements)

## Features ✨

- **ESG-Specialized Agent**  
  LCEL-based chain with tool selection for:
  - Elasticsearch document retrieval (📄 Corporate filings)
  - [Opoint News API](https://api-docs.opoint.com/) integration (📰 ESG news)
  - Multi-step reasoning with [IBM Granite-3.1 LLM](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)

- **Enterprise-Ready Infrastructure**  
  - Async implementation with Python 3.12
  - Docker containerization
  - API key security
  - Session management via `RunnableWithMessageHistory`

- **IBM Watsonx Integration**  
  - 🧠 [Granite-3.1](https://www.ibm.com/granite/docs/models/granite/) LLM (13B parameters)
  - 📥 [Slate-30M](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-slate-30m-english-rtrvr-v2-model-card.html?context=wx) English embeddings
  - Watson Assistant extension support

## Deployment 🛠️
### Local

```bash
git clone git@github.com:milad-o/Agentic-ESG-RAG.git
cd Agentic-ESG-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```
### Docker

```bash
git clone git@github.com:milad-o/Agentic-ESG-RAG.git
cd Agentic-ESG-RAG

# Build the Docker image
docker build -t esg-agent .

# Run the Docker container
docker run -p 4050:4050 esg-agent
```
> [!NOTE]
> To build the docker file, you need to configure the .env file.

## Configuration 🔧
Create a `.env` file in the root directory of the project and assign the proper values in the `.env.example` file.

## API Documentation 📚
### Endpoints
- **`POST /query`**: Main agent endpoint (API key required).
- **`GET /docs`**: Interactive [Swagger UI](https://swagger.io/tools/swagger-ui/) for API documentation.

### Example Request
```bash
curl -X POST "http://localhost:4050/query" \
  -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What ESG risks does Company X face according to their latest report?"}'
```
> [!NOTE]
> For cloud deployment, you obviouly need to replace the `localhost` with the proper domain.

## Technology Stack 🛠️
- **Core AI**
    - [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel/): Agent framework
    - [IBM Watsonx.ai](https://www.ibm.com/products/watsonx-ai): Granite-3.1 and Slate-30M
    - [Elasticsearch](https://www.elastic.co/elasticsearch/): Vector store and ELSER embeddings
- Infrastructure
    - [FastAPI](https://fastapi.tiangolo.com/): API framework
    - [Uvicorn](https://www.uvicorn.org/): ASGI server
    - [Docker](https://www.docker.com/): Containerization
    - [IBM Cloud Object Storage](https://www.ibm.com/products/cloud-object-storage): Object storage
    - [IBM Watsonx Assistant](https://www.ibm.com/products/watsonx-assistant): Chatbot and frontend

## Watsonx Assistant Extension
In order to use the app with Watsonx Assistant, you can use the `agent-assistant-ext.json` file. This file contains the necessary configuration to integrate the agent with Watsonx Assistant. It follows [OpenAPI 3.0.0](https://spec.openapis.org/oas/v3.0.0.html) specification.

> [!NOTE] 
> Please note that the server URL should be replaced with the actual URL of your FastAPI server in the `server.url` field.

## Acknowledgements 🙏
Part of this project was funded by a [Mitacs](https://www.mitacs.ca/) project as part of collaboration between [Rel8ed Analytics](https://www.rel8ed.to/) and Brock University to advance AI applications in the ESG space.