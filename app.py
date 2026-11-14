import os
import re
import glob
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="💼", layout="centered")
st.title("Zyro Dynamics HR Help Desk")
st.caption("Ask questions about Zyro Dynamics Pvt. Ltd. HR policies.")

groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not groq_key:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_key
if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "zyro-rag-challenge")

@st.cache_resource
def build_retriever(pdf_folder="zyro-dynamics-hr-corpus"):
    pdf_files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    docs = []
    for f in pdf_files:
        loaded = PyPDFLoader(f).load()
        for d in loaded:
            d.metadata["source_file"] = os.path.basename(f)
        docs.extend(loaded)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=350)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.75})

@st.cache_resource
def get_model_name(api_key: str) -> str:
    client = Groq(api_key=api_key)
    models = [m.id for m in client.models.list().data if "whisper" not in m.id and "tts" not in m.id and "guard" not in m.id]
    preferred = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    return next((m for m in preferred if m in models), models[0])

retriever = build_retriever()
llm = ChatGroq(model_name=get_model_name(groq_key), temperature=0.0, api_key=groq_key)

prompt = ChatPromptTemplate.from_template(
    "You are the HR Help Desk Assistant for Zyro Dynamics Pvt. Ltd.\n"
    "Answer ONLY using the provided HR policy context.\n"
    "If the question is outside Zyro Dynamics HR policies or not answered in the context, refuse politely and clearly.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Source Citations"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

if user_q := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        docs = retriever.invoke(user_q)
        context_text = "\n\n".join(
            f"[{d.metadata.get('source_file', 'Policy')} - Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
            for d in docs
        )
        raw_ans = (prompt | llm | StrOutputParser()).invoke({"context": context_text, "question": user_q})
        clean_ans = re.sub(r"<think>.*?</think>", "", raw_ans, flags=re.DOTALL).strip()
        st.markdown(clean_ans)

        citations = list(dict.fromkeys(
            f"{d.metadata.get('source_file', 'Policy')} (Page {d.metadata.get('page', 0) + 1})"
            for d in docs
        ))
        with st.expander("Source Citations"):
            for c in citations:
                st.write(f"- {c}")

        st.session_state.messages.append({"role": "assistant", "content": clean_ans, "sources": citations})