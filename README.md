# Financial RAG Agent with LangGraph & Pathway

An intelligent financial assistant built using **Flask**, **LangChain**, and **LangGraph**. This application routes user queries between a specialized financial vectorstore (containing SEC filings) and a web search engine to provide accurate, data-driven answers.

## 🚀 Features

* **Intelligent Routing:** Automatically determines if a query requires internal financial documents or a live web search.
* **Graph-Based Workflow:** Utilizes LangGraph to manage state, retrieval grading, and query transformation for high-accuracy RAG.
* **Pathway Integration:** Leverages `PathwayVectorClient` and `PathwayRetriever` for efficient indexing and retrieval of large-scale financial datasets.
* **Hybrid Data Sourcing:** Combines internal financial data (SEC filings) with real-time web search capabilities.
* **Interactive Web UI:** Includes a Flask-based chat interface (`medai.html`) for seamless user interaction.

## 🛠️ Tech Stack

* **Backend:** Flask
* **Orchestration:** LangGraph & LangChain
* **LLM:** OpenAI GPT-3.5 Turbo
* **Vector Database:** Pathway
* **Schema & Validation:** Pydantic

## 🏗️ Workflow Architecture

The system follows a sophisticated decision-making graph:
1.  **START:** Question is received.
2.  **Router:** LLM decides between `vectorstore` or `web_search`.
3.  **Retrieve/Search:** Fetches data from Pathway or the web.
4.  **Grade:** Evaluates the relevance of retrieved documents.
5.  **Transform:** If data is insufficient, the query is rephrased for a better search.
6.  **Generate:** Final answer is synthesised and returned to the UI.
---
# 🚀 Future Improvements & Roadmap

### 🛠️ Technical Enhancements
* **Advanced Reranking:** 
* **Async Workflow Execution:** 
* **Support for Multi-Modal Data:**
* **Memory & Conversation Context:** 
* **Human-in-the-loop (HITL):** 
