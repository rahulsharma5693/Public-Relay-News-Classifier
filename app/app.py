import shutil
import os
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
from typing import Literal
from pydantic import BaseModel, Field
import psycopg2
import pandas as pd

# if "chroma_ingested" not in st.session_state:
#     st.session_state["chroma_ingested"] = False

st.image("pr.png", width = 80)

# Streamlit UI
st.title("Public Relay News Relevance Classifier")

st.subheader("🧠 Choose LLM Provider")

model_provider = st.selectbox("🧠 Choose model provider:", [
    "OpenAI (cloud)",
    "Ollama (local)",
    "Ollama (remote)"
])

if model_provider == "Ollama (remote)":
    ollama_remote_url = st.text_input(
        "🌐 Enter remote Ollama base URL (e.g., http://192.168.1.50:11434)",
    )
    if not ollama_remote_url:
        st.warning("Please enter a valid base URL to proceed.")

if model_provider in ["OpenAI (cloud)"]:
    api_key = st.text_input(f"🔑 Enter your {model_provider.split()[0]} API key", type="password")
    if not api_key:
        st.warning("Please enter your API key to proceed.")

def build_full_prompt(df, outlet, title, body):
    examples = []
    for _, row in df.iterrows():
        examples.append(f"Outlet: {row['outlet_name']}\nTitle: {row['orig_title']}\nBody: {row['orig_body']}\nLabel: {row['relevance']}")

    few_shot_block = "\n\n".join(examples)

    criteria = """
An article is IRRELEVANT if it is a press release, stock price mention, alumni mention, sponsored content, AI-generated, read-more link, company brief, market talk, news roundup, newsletter, obituary, 8K filing, Dow Jones Market Talk, or Reuters brief.
All other articles are RELEVANT.
"""

    query_block = f"Outlet: {outlet}\nTitle: {title}\nBody: {body}\nLabel:"

    return f"""You are a news relevance classifier.

{criteria}

Here are 10 labeled examples:

{few_shot_block}

Now classify the following article:

{query_block}

Respond with one word only: RELEVANT or IRRELEVANT."""

def run_ingestion(Embeddings_model = None):

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        dbname="relevance_db",
        user="postgres",
        password=5693
    )
    df = pd.read_sql("SELECT * FROM relevance_examples;", conn)
    conn.close()

    df['combined'] = "<outlet_name>" + df['outlet_name'] + "</outlet_name>" +  "<title>" + df['orig_title'] + "</title>" + "<body>" + df['orig_body'] + "</body>"

    docs = []
    for _, row in df.iterrows():
        docs.append(Document(
            page_content=row["combined"],
            metadata={
                "analysis_id": row["analysis_id"],
                "outlet_name": row["outlet_name"],
                "orig_title": row["orig_title"],
                "orig_body": row["orig_body"],
                "relevance": row["relevance"]
            }
        ))

    embeddings = Embeddings_model
    vectorstore = Chroma(
        collection_name="relevance_examples",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    existing_count = vectorstore._collection.count()
    expected_count = len(docs)

    return docs, vectorstore, existing_count, expected_count

@st.cache_data
def load_pg_examples():
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        dbname="relevance_db",
        user="postgres",
        password="5693"
    )
    df = pd.read_sql("SELECT * FROM relevance_examples ORDER BY RANDOM() LIMIT 10;", conn)
    conn.close()
    df.columns = [col.lower().strip() for col in df.columns]
    return df



if model_provider == "Ollama (local)":
    llm = ChatOllama(model="qwen3:1.7b", base_url=os.environ["LANGCHAIN_OLLAMA_URL"])
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url = os.environ["LANGCHAIN_OLLAMA_URL"])
elif model_provider == "Ollama (remote)":
    llm = ChatOllama(model="qwen3:1.7b", base_url=ollama_remote_url)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_remote_url)
elif model_provider == "OpenAI (cloud)" and api_key:
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
else:
    st.stop()
  # Prevent app from running without valid model setup
# Data model
class predict(BaseModel):
    """Route a user query to the most relevant datasource."""

    prediction: Literal["RELEVANT", "IRRELEVANT"] = Field(
        ...,
        description="Given a user statement tell if the statement is RELEVANT or IRRELEVANT ",
    )

llm_predict = llm.with_structured_output(predict)

vectorstore = Chroma(
    collection_name="relevance_examples",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LangGraph state
from typing import TypedDict, List

class RelevanceState(TypedDict):
    outlet: str
    title: str
    body: str
    retrieved_examples: List[Document]
    decision: str
    result: str
    decide: str
    combined: str

# Decision node
def decide_node(state: RelevanceState):
    prompt = f"""
You are a relevance classification agent deciding whether to classify a news article directly or use similar examples (retrieved via semantic search).

Use RAG if:
- The article is ambiguous or context-dependent
- It contains subtle cues (e.g., alumni mentions, press releases, market talk)
- You are unsure whether it fits the RELEVANT or IRRELEVANT criteria

Use DIRECT if:
- The article clearly fits known patterns (e.g., sponsored content, earnings briefs, AI breakthroughs)
- You are confident in your classification without examples

Here is the article:

Outlet: {state['outlet']}
Title: {state['title']}
Body: {state['body']}

Should you use RAG or DIRECT?

Respond with one word only: RAG or DIRECT.
"""
    # Data model
    class classify(BaseModel):
        """Route a user query to the most relevant datasource."""

        prediction: Literal["RAG", "DIRECT"] = Field(
            ...,
            description="Given a user statement tell whether to classify a news article directly(DIRECT) or use similar examples(RAG).",
        )
    decision = llm.with_structured_output(classify).invoke(prompt)
    print(f"decision node: {decision.prediction}")
    return {"decide":"retrieve"} if decision.prediction == "RAG" else {"decide":"DIRECT"}

# Retrieval node
def retrieve_node(state: RelevanceState) -> RelevanceState:
    docs = retriever.invoke(state["combined"])
    state["retrieved_examples"] = docs
    return state

# Classification node
def classify_node(state: RelevanceState) -> RelevanceState:
    if not state.get("retrieved_examples"):
        # DIRECT mode — use full prompt from PostgreSQL
        df = load_pg_examples()
        prompt = build_full_prompt(df, state["outlet"], state["title"], state["body"])
    else:
        # RAG mode — use retrieved examples
        examples = ""
        for doc in state["retrieved_examples"]:
            meta = doc.metadata
            examples += f"""Outlet: {meta['outlet_name']}
Title: {meta['orig_title']}
Body: {doc.page_content}
Label: {meta['relevance']}\n\n"""

        criteria = """
An article is IRRELEVANT if it is a press release, stock price mention, alumni mention, sponsored content, AI-generated, read-more link, company brief, market talk, news roundup, newsletter, obituary, 8K filing, Dow Jones Market Talk, or Reuters brief.
All other articles are RELEVANT.
"""

        query = f"""Outlet: {state['outlet']}\nTitle: {state['title']}\nBody: {state['body']}\nLabel:"""

        prompt = f"""You are a news relevance classifier.

{criteria}

Here are some labeled examples:

{examples}

Now classify the following article:

{query}

Respond with one word only: RELEVANT or IRRELEVANT."""

    result = llm_predict.invoke(prompt)
    state["result"] = result.prediction
    return state


def route_node(state:RelevanceState):

    if state["decide"] == "retrieve":
        return "retrieve"
    else:
        return "classify"

# LangGraph workflow
workflow = StateGraph(RelevanceState)
workflow.add_node("decide", RunnableLambda(decide_node))
workflow.add_node("retrieve", RunnableLambda(retrieve_node))
workflow.add_node("classify", RunnableLambda(classify_node))

workflow.set_entry_point("decide")
workflow.add_conditional_edges("decide", route_node, {
    "retrieve": "retrieve",
    "classify": "classify"
})
workflow.add_edge("retrieve", "classify")
workflow.add_edge("classify", END)

graph = workflow.compile()
# graph

st.subheader("📥 Ingest Labeled Examples into ChromaDB")

if "chroma_ingested" not in st.session_state:
    st.session_state["chroma_ingested"] = False

if st.button("Start Ingestion"):
    with st.spinner("Connecting to PostgreSQL and preparing documents..."):
        docs, vectorstore, existing_count, expected_count = run_ingestion(embeddings)

    if existing_count >= expected_count:
        st.session_state["chroma_ingested"] = True
        st.info("✅ ChromaDB already contains all ingested documents. Skipping ingestion.")
        st.rerun()
    else:
        progress_bar = st.progress(0)
        for i, doc in enumerate(docs):
            vectorstore.add_documents([doc])
            progress_bar.progress((i + 1) / expected_count)

        st.session_state["chroma_ingested"] = True
        st.success(f"✅ Ingested {expected_count} documents into ChromaDB.")
        st.rerun()

if st.button("🧹 Reset ChromaDB"):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        st.success("ChromaDB folder deleted.")
        st.rerun()


st.subheader("🧠 Choose Classification Mode")

if not st.session_state["chroma_ingested"]:
    st.radio("Mode", ["Direct"], index=0, disabled=True)
    st.markdown("⚠️ *RAG and Agentic modes are disabled until labeled examples are ingested into ChromaDB.*")
    st.markdown("""
    - **RAG** uses semantic search to retrieve similar examples for ambiguous articles.
    - **Agentic** uses a decision node to choose between Direct and RAG based on article complexity.
    """)
    mode = "Direct"
else:
    mode = st.radio("Mode", ["Direct", "RAG", "Agentic (LangGraph)"], index=0)


st.markdown("---")
st.subheader("Classification via CSV")

uploaded_file = st.file_uploader("Upload a CSV file with 4 columns: Analysis_id, Outlet_name, Orig_title, Orig_body", type=["csv"])

# outlet = st.text_input("Outlet Name")
# title = st.text_input("Article Title")
# body = st.text_area("Article Body", height=200)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ["analysis_id", "outlet_name", "orig_title", "orig_body"]
    df.columns = [col.lower().strip() for col in df.columns]  # Normalize headers

    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV must contain lowercase columns: analysis_id, outlet_name, orig_title, orig_body")
    else:
        df['combined'] = "<outlet_name>" + df['outlet_name'] + "</outlet_name>" +  "<title>" + df['orig_title'] + "</title>" + "<body>" + df['orig_body'] + "</body>"
        st.success(f"✅ Loaded {len(df)} records. Ready to classify.")
    # Proceed with classification...
        if st.button("Run News Classification"):
            results = []

            for _, row in df.iterrows():
                state = {
                    "outlet": row["outlet_name"],
                    "title": row["orig_title"],
                    "body": row["orig_body"],
                    "combined": row["combined"]
                }

                if mode == "Direct":
                    # labeled_df = load_labeled_examples()
                    result = classify_node(state)
                elif mode == "RAG":
                    state = retrieve_node(state)
                    result = classify_node(state)
                else:
                    result = graph.invoke(state)

                results.append({
                    "Analysis_id": row["analysis_id"],
                    "Relevance": result["result"]
                })

            output_df = pd.DataFrame(results)
            relevant_count = output_df['Relevance'].value_counts().get("RELEVANT", 0)
            irrelevant_count = output_df['Relevance'].value_counts().get("IRRELEVANT", 0)
            st.write(f"✅ RELEVANT: {relevant_count}, ❌ IRRELEVANT: {irrelevant_count}")
            st.dataframe(output_df)

            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", data=csv, file_name="classified_results.csv", mime="text/csv")
