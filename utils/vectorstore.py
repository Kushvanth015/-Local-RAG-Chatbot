from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings


# -------------------------
# Split documents into chunks
# -------------------------
def _split_docs(documents):
    """
    Chunking settings tuned for:
    - short topics (BPE, PPO, RLHF)
    - definitions
    - interview questions
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,       # slightly bigger = better context per chunk
        chunk_overlap=150,    # overlap helps not to miss definitions
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# -------------------------
# Embeddings
# -------------------------
def _embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


# -------------------------
# Build new vectorstore
# -------------------------
def build_vectorstore(documents):
    chunks = _split_docs(documents)
    embeddings = _embeddings()
    return FAISS.from_documents(chunks, embeddings)


# -------------------------
# Add new docs into existing store
# -------------------------
def add_docs_to_vectorstore(vectorstore, documents):
    chunks = _split_docs(documents)
    vectorstore.add_documents(chunks)
    return vectorstore


# -------------------------
# Save / Load FAISS index
# -------------------------
def save_vectorstore(vectorstore, path="faiss_index"):
    vectorstore.save_local(path)


def load_vectorstore(path="faiss_index"):
    embeddings = _embeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
