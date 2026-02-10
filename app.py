import os
import re
import json
import shutil
import hashlib
import numpy as np
import streamlit as st

from langchain_community.embeddings import OllamaEmbeddings

from utils.loader import load_uploaded_files
from utils.vectorstore import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    add_docs_to_vectorstore
)
from utils.rag_chain import create_rag_chain


INDEX_PATH = "faiss_index"
USERS_FILE = "users.json"


# -------------------------
# Helpers
# -------------------------
def format_numbered_output(text: str) -> str:
    text = re.sub(r"\s(?=\d+\.)", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def login_ui():
    st.title("🔐 Login")
    st.caption("Login to access the SHOP-ASSIST-RAG-Chatbot.")

    users = load_users()

    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password").strip()

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.is_logged_in = True
            st.session_state.username = username
            st.session_state.role = users[username].get("role", "user")
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


def get_index_id():
    """
    Generates a stable ID for the current index folder.
    This helps store memory PER INDEX.
    """
    if not os.path.exists(INDEX_PATH):
        return "no_index"
    return hashlib.md5(INDEX_PATH.encode()).hexdigest()


def chat_history_to_text(messages, max_turns=8):
    """
    Converts Streamlit chat memory into plain text.
    LangChain prompt expects chat_history as string.
    """
    history = []
    for msg in messages[-max_turns:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {msg['content']}")
    return "\n".join(history)

def memory_to_tuples(messages, max_turns=8):
    """
    Converts Streamlit memory into:
    [("user msg", "assistant msg"), ...]
    This is the required format for ConversationalRetrievalChain.
    """
    pairs = []
    user_msg = None

    for msg in messages[-(max_turns * 2):]:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg is not None:
            pairs.append((user_msg, msg["content"]))
            user_msg = None

    return pairs



# -------------------------
# Embeddings (for score)
# -------------------------
emb_model = OllamaEmbeddings(model="nomic-embed-text")


def groundedness_score(answer, sources, top_k=2):
    """
    Score = similarity between answer and top retrieved chunks (0 to 100)
    Higher = more grounded.
    """
    if not sources or not answer.strip():
        return 0.0

    top_sources = sources[:top_k]
    source_text = " ".join([doc.page_content for doc in top_sources])

    # Safe truncation to avoid embedding overflow
    source_text = source_text[:2500]

    ans_vec = np.array(emb_model.embed_query(answer))
    src_vec = np.array(emb_model.embed_query(source_text))

    sim = np.dot(ans_vec, src_vec) / (np.linalg.norm(ans_vec) * np.linalg.norm(src_vec))
    return round(float(sim) * 100, 2)


# -------------------------
# Session State Init
# -------------------------
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "role" not in st.session_state:
    st.session_state.role = "user"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Memory per index
if "doc_memory" not in st.session_state:
    st.session_state.doc_memory = {}


# -------------------------
# LOGIN GATE
# -------------------------
if not st.session_state.is_logged_in:
    login_ui()
    st.stop()


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="SHOP-ASSIST-RAG-Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 SHOP-ASSIST-RAG-Chatbot")
st.caption("Chat with PDFs/TXTs using Local RAG (LangChain + FAISS + Ollama).")

st.sidebar.success(f"👤 Logged in as: {st.session_state.username} ({st.session_state.role})")

if st.sidebar.button("🚪 Logout"):
    st.session_state.is_logged_in = False
    st.session_state.username = None
    st.session_state.role = "user"
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.rerun()


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Controls")

strict_mode = st.sidebar.toggle("🔒 Strict Document Mode", value=True)

developer_mode = False
debug_retriever = False

if st.session_state.role == "admin":
    developer_mode = st.sidebar.toggle("🛠 Developer Mode", value=False)
    if developer_mode:
        debug_retriever = st.sidebar.toggle("🪲 Retriever Debug", value=False)

st.sidebar.markdown("---")


# -------------------------
# Load saved index if exists
# -------------------------
if st.session_state.vectorstore is None and os.path.exists(INDEX_PATH):
    st.sidebar.info("Found saved FAISS index. Loading...")
    st.session_state.vectorstore = load_vectorstore(INDEX_PATH)
    st.session_state.qa_chain = create_rag_chain(st.session_state.vectorstore, strict_mode)
    st.sidebar.success("✅ Loaded saved index!")


# -------------------------
# Admin-only Index Management
# -------------------------
if st.session_state.role == "admin":

    if st.sidebar.button("🧹 Clear Chat (This Index)"):
        index_id = get_index_id()
        st.session_state.doc_memory[index_id] = []
        st.rerun()

    if st.sidebar.button("🗑️ Delete Saved Index"):
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)

        st.session_state.vectorstore = None
        st.session_state.qa_chain = None

        index_id = get_index_id()
        st.session_state.doc_memory[index_id] = []

        st.sidebar.success("Deleted saved FAISS index.")
        st.rerun()

    st.sidebar.markdown("---")

    uploaded_files = st.file_uploader(
        "📄 Upload multiple PDF/TXT files (Admin only)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📌 Add Documents to Index"):
            with st.spinner("Loading documents..."):
                docs = load_uploaded_files(uploaded_files)

            with st.spinner("Updating FAISS index..."):
                if os.path.exists(INDEX_PATH):
                    st.session_state.vectorstore = load_vectorstore(INDEX_PATH)
                    st.session_state.vectorstore = add_docs_to_vectorstore(
                        st.session_state.vectorstore, docs
                    )
                else:
                    st.session_state.vectorstore = build_vectorstore(docs)

            with st.spinner("Saving updated index..."):
                save_vectorstore(st.session_state.vectorstore, INDEX_PATH)

            st.session_state.qa_chain = create_rag_chain(
                st.session_state.vectorstore, strict_mode
            )

            st.success("✅ Index updated! Old PDFs + New PDFs are now stored.")

else:
    st.info("📌 Only Admin can upload documents. Please contact Admin to update the index.")


# -------------------------
# Memory per index
# -------------------------
index_id = get_index_id()
if index_id not in st.session_state.doc_memory:
    st.session_state.doc_memory[index_id] = []


# -------------------------
# Chat UI
# -------------------------
st.markdown("## 💬 Chat")

for msg in st.session_state.doc_memory[index_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
# Ask question
# -------------------------
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask something about your documents...")

    if user_input:
        # Save user message
        st.session_state.doc_memory[index_id].append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_text = chat_history_to_text(
                    st.session_state.doc_memory[index_id],
                    max_turns=8
                )
                chat_history = memory_to_tuples(
                    st.session_state.doc_memory[index_id],
                    max_turns=8
                )
                result = st.session_state.qa_chain({
                    "question": user_input,
                    "chat_history": chat_history
               })

            answer = result.get("answer", "").strip()
            sources = result.get("source_documents", [])



            if not answer:
                answer = "I don't know based on the document."

            answer = format_numbered_output(answer)

            # ✅ Only calculate score in developer mode
            score = 0
            if developer_mode:
                score = groundedness_score(answer, sources)
            # Strict enforcement (only if developer mode OR you want it always)
            if strict_mode and developer_mode and score < 35:
                answer = "I don't know based on the document."

            # -------------------------
            # PRODUCTION OUTPUT
            # -------------------------
            st.markdown(answer)

            # -------------------------
            # ADMIN Developer Output
            # -------------------------
            if developer_mode:

                if score < 40:
                    st.warning("⚠️ Low groundedness score. Answer may not be fully supported.")

                # Retriever debug
                if debug_retriever:
                    st.markdown("### 🪲 Retriever Debug (Retrieved Chunks)")
                    if sources:
                        for i, doc in enumerate(sources, 1):
                            meta = doc.metadata
                            file_name = meta.get("source_file", "Unknown File")
                            page = meta.get("page", "N/A")
                            st.markdown(f"**Chunk {i} | File:** `{file_name}` | **Page:** `{page}`")
                            st.write(doc.page_content[:500] + "...")
                    else:
                        st.warning("Retriever returned 0 chunks.")
                    st.markdown("---")

                # Evidence
                st.markdown("### 🧾 Evidence (Top 2 chunks)")
                if sources:
                    for i, doc in enumerate(sources[:2], 1):
                        text = doc.page_content.strip().replace("\n", " ")
                        st.write(f"**Evidence {i}:** {text[:500]}...")
                else:
                    st.info("No evidence retrieved.")

                # Score
                st.markdown("### 🧠 Groundedness Score")
                st.progress(int(score))
                st.write(f"**{score}/100**")

                # Sources
                with st.expander("📌 Sources (All chunks)"):
                    if sources:
                        for i, doc in enumerate(sources, 1):
                            meta = doc.metadata
                            file_name = meta.get("source_file", "Unknown File")
                            page = meta.get("page", "N/A")
                            st.markdown(f"**Source {i} | File:** `{file_name}` | **Page:** `{page}`")
                            st.write(doc.page_content[:800] + "...")
                    else:
                        st.info("No sources retrieved.")

        # Save assistant response into memory
        st.session_state.doc_memory[index_id].append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("No index found. Admin must upload documents and build the FAISS index.")
