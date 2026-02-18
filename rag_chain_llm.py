from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI
from pydantic import BaseModel

import os

os.environ["GOOGLE_API_KEY"] = ""

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

# FAISS + embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Tum ek helpful AI ho.
Neeche diye gaye context se hi jawab dena.
Agar jawab context me na ho to "Mujhe is document me jawab nahi mila" likho.

Context:
{context}

Question:
{question}

Answer (English):
""")

def rag_chain():
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    return chain
app = FastAPI(title="RAG PDF Chat API")
app = FastAPI(debug=True)
class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    chain = rag_chain()
    response = chain.invoke(query.question)
    return {"answer": response.content}
