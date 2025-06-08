from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ( 
    WebBaseLoader,
    PyPDFLoader,
    TextLoader
)
import time

t = time.time()

def load_documents(source:str, source_type:str):
    if source_type =="web":
        loader = WebBaseLoader(web_paths=[source])
    elif source_type == "pdf":
        loader = PyPDFLoader(file_path=source)
    elif source_type == "text":
        loader = TextLoader(file_path=source)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    print(time.time() - t)
    return retriever
