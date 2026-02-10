from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


def create_rag_chain(vectorstore, strict_mode=True):
    llm = Ollama(
        model="qwen2.5:3b",
        temperature=0.1,
        num_predict=150
    )

    # Retriever
    if strict_mode:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

    # Prompt
    strict_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""
You are a strict document assistant.

Rules:
1) Answer ONLY using the given context.
2) If the answer is not present in the context, say:
   "I don't know based on the document."
3) Do not add extra knowledge.
4) Keep answers short and clear.

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": strict_prompt}
    )

    return qa_chain
