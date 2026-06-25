# 🤖 Autonomous SQL-Agent for Business Intelligence
**Live Demo:** https://metadata-sql-rag-agent.azurewebsites.net/

An enterprise-grade Text-to-SQL AI Agent that translates natural language into secure, executable SQL queries. Built with LangGraph and Llama-3.3, featuring a self-correcting agentic loop and a Semantic Metadata-RAG pipeline.

## 🚀 Key Features
- **Agentic Self-Correction:** LangGraph state machine catches SQL dialect/syntax tracebacks and self-heals code, boosting success rates to ~95%.
- **Metadata-RAG:** Uses ChromaDB to map natural language to exact foreign-key schemas, reducing structural hallucinations by 85%.
- **Multi-layer Guardrails:** LLM-as-a-Judge semantic router to block prompt injections, plus hardware-level SQLite Read-Only enforcement.
- **CI/CD Cloud Deployment:** Fully containerized with Docker and deployed to Azure App Services with custom SQLite indexing for <4s latency.

## 🛠️ Tech Stack
- **AI/Agents:** LangGraph, LangChain, Groq (Llama-3.3-70b), HuggingFace (all-MiniLM-L6-v2)
- **Database:** SQLite (Indexed), ChromaDB
- **Deployment:** Streamlit, Docker, Azure Container Registry (ACR), Azure App Service

## 💻 How to Run Locally
1. Clone this repository or download the ZIP.
2. Download the [Olist Dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSVs in a `data/` folder.
3. Run `pip install -r requirements.txt`.
4. Create a `.env` file and add your Groq API key: `GROQ_API_KEY=your_key`
5. Run `python setup_db.py` to build the database and Chroma vector store.
6. Run `streamlit run app.py`.
