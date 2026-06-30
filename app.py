import streamlit as st
import sqlite3
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
# --- AUTO-BUILD DATABASE FOR CLOUD DEPLOYMENT ---
if not os.path.exists("olist.db") or not os.path.exists("chroma_db"):
    with st.spinner("⚙️ Building Database & Vector Store for the first time (Takes ~60 seconds)..."):
        import setup_db
        setup_db.build_sqlite_db()
        setup_db.build_metadata_rag()
        st.success("Database built successfully! Refreshing...")
        st.rerun()
st.set_page_config(page_title="Autonomous SQL Agent", layout="wide", page_icon="🤖")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("API Key not found! Please set GROQ_API_KEY.")
    st.stop()
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

class AgentState(TypedDict):
    question: str
    chat_history: str 
    schema_context: str
    sql_query: str
    sql_error: str
    query_result: str
    final_answer: str
    retries: int
    is_off_topic: bool

def retrieve_and_guardrail(state: AgentState):
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict security router for an e-commerce database. "
                   "If the prompt is about data analytics (sales, orders, customers, products, reviews, revenue), reply 'YES'. "
                   "If it is a general question, math (e.g. 1+2), coding help, or greeting, reply 'NO'. "
                   "Reply ONLY with YES or NO."),
        ("human", "{question}")
    ])
    
    intent = (judge_prompt | llm).invoke({"question": state["question"]}).content.strip().upper()
    
    if "NO" in intent:
        return {"is_off_topic": True, "schema_context": "", "retries": 0}

    docs = vectorstore.similarity_search(state["question"], k=5)
    schemas = [doc.metadata["schema"] for doc in docs]
    return {"schema_context": "\n\n".join(schemas), "retries": state.get("retries", 0), "is_off_topic": False}

def generate_sql(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert SQLite data analyst. Write a raw SQL query to answer the user's question. "
                   "Return ONLY the SQL code, no markdown. "
                   "CRITICAL BUSINESS LOGIC: "
                   "1. NEVER use arbitrary table aliases like T1, T2, T3. ALWAYS write out the full table name (e.g., order_items.price). "
                   "2. Revenue is calculated as SUM(order_items.price). There is NO 'quantity' column. "
                   "3. NEVER join order_id directly to product_id or seller_id. You MUST walk through the 'order_items' table. "
                   "4. If asked for an overall metric (e.g. 'average across all'), return ONE aggregate row. Do NOT use GROUP BY. "
                   "5. Date math must use: julianday(date1) - julianday(date2). "
                   "6. Always use LIMIT 50. "
                   "If you receive a previous error, FIX your SQL based on the traceback.\n\n"
                   "Relevant Schemas:\n{schema_context}\n\n"
                   "Chat History:\n{chat_history}\n\n"
                   "Previous Error:\n{sql_error}"),
        ("human", "{question}")
    ])
    
    sql_query = (prompt | llm).invoke({
        "schema_context": state["schema_context"],
        "chat_history": state["chat_history"],
        "sql_error": state.get("sql_error", "None"),
        "question": state["question"]
    }).content.replace('```sql', '').replace('```', '').strip()
    
    return {"sql_query": sql_query}

def execute_sql(state: AgentState):
    try:
        conn = sqlite3.connect("file:olist.db?mode=ro", uri=True)
        cursor = conn.cursor()
        
        sql_lower = state["sql_query"].lower()
        if re.search(r'\b(drop|delete|update|insert|alter|create|truncate)\b', sql_lower):
            raise Exception("SECURITY ALERT: Modifying the database is strictly prohibited.")

        cursor.execute(state["sql_query"])
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchmany(50) 
        formatted_results = [dict(zip(columns, row)) for row in results]
        conn.close()
        return {"query_result": str(formatted_results), "sql_error": ""}
    except Exception as e:
        return {"sql_error": str(e), "retries": state["retries"] + 1}

def generate_answer(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and professional BI Analyst. The provided Data is the exact result of a SQL query. Trust the Data. "
                   "CRITICAL RULE: NEVER perform manual arithmetic (addition, division, etc.) on the data. "
                   "1. Answer the user's question using a clear, conversational full sentence. "
                   "2. If the data contains multiple rows (like a 'Top 5' list), format it nicely as a bulleted list including the names/IDs and their corresponding values. "
                   "3. If the data is empty, politely state that no records were found. "
                   "Do not mention SQL, tracebacks, or databases in your response."),
        ("human", "Question: {question}\nData: {query_result}")
    ])
    answer = (prompt | llm).invoke({"question": state["question"], "query_result": state["query_result"]}).content
    return {"final_answer": answer}

def route_start(state: AgentState):
    return "end" if state["is_off_topic"] else "generate_sql"

def route_after_execution(state: AgentState):
    if state["sql_error"] and state["retries"] < 3:
        return "generate_sql" 
    elif state["sql_error"]:
        return "end" 
    return "generate_answer"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve_and_guardrail", retrieve_and_guardrail)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve_and_guardrail")
workflow.add_conditional_edges("retrieve_and_guardrail", route_start, {"generate_sql": "generate_sql", "end": END})
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", route_after_execution, {"generate_sql": "generate_sql", "generate_answer": "generate_answer", "end": END})
workflow.add_edge("generate_answer", END)
app_graph = workflow.compile()

with st.sidebar:
    st.header("⚙️ Architecture Under the Hood")
    st.markdown("1. **Semantic Metadata-RAG:** Defeats hallucinations using ChromaDB.\n2. **Agentic Self-Correction:** A LangGraph state machine catches SQL errors.\n3. **Multi-layer Guardrails:** Read-Only enforcement and topical routing.")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Autonomous Business Intelligence Agent")
st.markdown("*Talk to your database securely. Powered by Llama-3 & LangGraph.*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] != "system": 
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about the data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
    
    with st.spinner("Executing Agentic Loop..."):
        result = app_graph.invoke({"question": prompt, "chat_history": chat_history, "retries": 0, "sql_error": ""})
        
        if result.get("is_off_topic"):
            response = "🛡️ **Guardrail Triggered:** Please ask me a question related to the company's data."
        elif result.get("sql_error"):
            response = f"⚠️ **Execution Failed:** Last error: `{result['sql_error']}`"
        else:
            response = result["final_answer"]
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        
        if not result.get("is_off_topic"):
            with st.expander("🔍 View Agent Execution Trace"):
                st.code(f"-- Generated SQL:\n{result.get('sql_query')}", language="sql")
                st.write(f"**Self-Correction Iterations required:** {result.get('retries')}")