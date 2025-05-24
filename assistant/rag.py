from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def load_doc_chunks(path):
    loader = PyPDFLoader(path)
    return loader.load_and_split()

def get_retriever(chunks):
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(chunks, embedding=embeddings)
    return db.as_retriever()

def get_rag_chain(retriever):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key="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")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_from_docs(question, rag_chain):
    return rag_chain.run(question)