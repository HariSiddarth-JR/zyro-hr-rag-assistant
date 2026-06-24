import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#-------------------------
#CONFIG
#-------------------------

LLM_MODEL = "llama-3.3-70b-versatile"

#Put your Groq API Key in Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
#"""-------------------------
#LOAD RAG COMPONENTS
#-------------------------"""

@st.cache_resource
def build_rag():

    loader = PyPDFDirectoryLoader("zyro-dynamics-hr-corpus")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm

retriever, llm = build_rag()

#"""-------------------------
#PROMPT
#-------------------------"""

RAG_PROMPT = ChatPromptTemplate.from_template(
'''
You are an HR assistant for Zyro Dynamics.

Answer using ONLY the provided context.

Provide a complete answer and include all relevant policy details found in the context.

If multiple requirements exist, summarize them as bullet points.

If the answer is not present, respond:
"Information not found in HR policies."

Context:
{context}

Question:
{question}

Answer:
'''
)

#"""-------------------------
#HELPERS
#-------------------------"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": context,
        "question": question
    })

REFUSAL_MESSAGE = (
"I can only answer questions related to Zyro Dynamics HR policies."
)

def ask_bot(question):

    hr_keywords = [
        "leave",
        "employee",
        "policy",
        "salary",
        "benefits",
        "work from home",
        "performance",
        "security",
        "travel",
        "expense",
        "onboarding",
        "separation",
        "conduct",
        "posh",
        "hr"
    ]

    if not any(k in question.lower() for k in hr_keywords):
        return REFUSAL_MESSAGE

    return rag_chain(question)
#"""-------------------------
#UI
#-------------------------"""

st.set_page_config(
page_title="Zyro HR Assistant",
page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Assistant")

question = st.text_input(
"Ask a question about HR policies"
)

if st.button("Submit"):

    if question.strip():

        with st.spinner("Searching policies..."):
            answer = ask_bot(question)

        st.markdown("### Answer")
        st.write(answer)
