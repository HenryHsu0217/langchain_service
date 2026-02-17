# LangChain Multi-Agent API with Google Cloud SQL + Google Drive

## 📌 Overview

This project implements a **FastAPI-based multi-agent system** using LangChain, designed to:

* Answer general questions (LLM-only assistant)
* Perform web search via Tavily
* Retrieve knowledge from documents stored in **Google Cloud SQL (Postgres)**
* Maintain **session-based conversational memory**

The system is deployed on a **Google Cloud VM (Compute Engine)** and works together with a separate ingestion pipeline that:

1. Reads PDF files from **Google Drive**
2. Cleans and processes them
3. Stores them into **Cloud SQL (Postgres)**

This application **does NOT directly access Google Drive** — it only queries the processed data stored in Postgres.

---

## 🧠 Architecture

### Data Flow

```
Google Drive (PDFs)
        ↓
[Ingestion Service - separate app]
        ↓
Google Cloud SQL (Postgres)
        ↓
[This Application]
        ↓
FAISS (in-memory vector store)
        ↓
LangChain Agents (RAG / Search / Hybrid)
        ↓
FastAPI Endpoints
```

---

## 🚀 Features

### 1. Xassistant (Basic LLM)

* Simple ChatGPT-like assistant
* No tools or external knowledge

### 2. Sagent (Search Agent)

* Uses **Tavily API** for web search
* Can fetch real-time information

### 3. Ragent (Retriever Agent)

* Queries documents stored in **Cloud SQL**
* Uses:

  * OpenAI embeddings (`text-embedding-3-large`)
  * FAISS vector store (in-memory)

### 4. RSagent (Hybrid Agent)

* Combines:

  * Web search (Tavily)
  * Document retrieval (Postgres)

### 5. Memory-enabled Agents

* Session-based chat memory
* Stored in-memory (per session ID)

---

## 🗂 Project Structure

```
.
├── main.py              # FastAPI app with agents and endpoints
├── loading_doc.py       # Loads data from Postgres and builds retriever tool
├── ai_with_memory.py    # Memory-enabled chat chain
├── requirements.txt
├── Dockerfile
```

---

## ⚙️ Environment Variables

Set the following environment variables before running:

```bash
export OPENAI_API_KEY=your_openai_key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

---

## 🗄️ Database Configuration

Configured in `loading_doc.py`:

```python
PROJECT_ID="langchain-service"
REGION="asia-east1"
INSTANCE="google-drive-vector"
DATABASE="Google-drive-files"
TABLE_NAME="document_test"
```

⚠️ **Important**: Move credentials like DB password to environment variables for security.

---

## 🧩 How Retrieval Works

1. Load documents from Postgres:

   ```python
   loader = PostgresLoader.create_sync(...)
   docs = loader.load()
   ```

2. Split into chunks:

   ```python
   CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   ```

3. Convert to embeddings:

   ```python
   OpenAIEmbeddings()
   ```

4. Store in FAISS:

   ```python
   FAISS.from_documents(...)
   ```

5. Expose as LangChain tool:

   ```python
   create_retriever_tool(...)
   ```

---

## 🌐 API Endpoints

### LangServe Routes

| Endpoint      | Description        |
| ------------- | ------------------ |
| `/Xassistant` | Basic assistant    |
| `/Sagent`     | Web search agent   |
| `/Ragent_db`  | DB retrieval agent |

---

### Custom Endpoints

#### Basic Assistant

```
POST /query/Xassitant
```

#### Assistant with Memory

```
POST /query/Xassistant-with-memory
POST /clear/Xassistant-with-memory
```

---

#### Search Agent

```
POST /query/Sagent
POST /query/Sagent-memory
POST /clear/Sagent-memory
```

---

#### Retriever Agent

```
POST /query/Ragent
POST /query/Ragent-memory
POST /clear/Ragent-memory
```

---

#### Hybrid Agent (Search + Retrieval)

```
POST /query/RSagent-memory
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t langchain-app .
```

### Run Container

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your_key \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
  langchain-app
```

---

## ☁️ Deployment (Google Cloud VM)

1. Create VM instance (Compute Engine)
2. Install Docker
3. Copy project files
4. Build and run container
5. Open port `8080` in firewall rules

---

## ⚠️ Known Limitations

* 🔁 FAISS is rebuilt **every time `create_db_tool()` is called**
* 🧠 Memory is stored **in RAM only** (not persistent)
* ⚡ Not optimized for large-scale data
* 🔐 Hardcoded DB credentials (should be fixed)

---

## 🔧 Suggested Improvements

* Use **persistent vector DB** (e.g., PGVector, Chroma, Weaviate)
* Cache FAISS index instead of rebuilding
* Replace in-memory history with Redis / DB
* Add authentication to API
* Fix minor bugs:

  * `{input}` vs `{question}` mismatch
  * returning `set()` instead of JSON
  * RSagent memory structure issue

---

## 🧪 Example Request

```bash
curl -X POST http://localhost:8080/query/Ragent \
-H "Content-Type: application/json" \
-d '{"question": "What is this document about?"}'
```

---

## 📌 Summary

This system is a **multi-agent RAG API** that:

* Uses LangChain agents
* Retrieves knowledge from Cloud SQL
* Combines search + retrieval
* Supports conversational memory
* Runs in Docker on Google Cloud VM

---

## 👨‍💻 Author Notes

Designed for experimentation with:

* LangChain agents
* RAG systems
* Multi-tool orchestration
* Cloud-based AI services

---

If you plan to scale this system, consider moving toward:

* **Vector-native databases**
* **Async pipelines**
* **Persistent memory backends**

---
