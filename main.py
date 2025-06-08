import streamlit as st
from langchain_groq import ChatGroq
from nodes import speak
from nodes.speech_recog import listen_to_user
from setups import load_documents
from nodes.retrieve import load_rag_chain
from pydub import AudioSegment
import io

retriever = None

st.set_page_config(page_title="Voice RAG Assistant", layout="centered")
st.title("🧠 RAG-powered AI Assistant")

option = st.selectbox("Choose an input method: ", ("Upload File", "Enter URL"))

if option == "Upload File":
    upload_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    if upload_file:
        with st.spinner("Loading Documents and building RAG Chain.."):
            retriever = load_documents(upload_file, "pdf")
elif option == "Enter URL":
    url = st.text_input("Enter URL: ")
    if url:
        with st.spinner("Loading Documents and building RAG Chain.."):
            retriever = load_documents(url, "web")

if retriever:
    llm = ChatGroq(model="llama-3.1-8b-instant")
    with st.spinner("Loading RAG Chain"):
        rag_chain = load_rag_chain(llm, retriever)

    audio_input_key = "audio_input"
    audio = st.audio_input("Ask query:", key=audio_input_key)

    if audio:
        audio = audio.read()
        audio = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        question = str(listen_to_user(wav_io))
        if "exit" in question.lower():
            st.success("Exiting")
        if question:
            with st.spinner("Generating response..."):
                res = rag_chain.invoke({"input": question})
                answer = str(res['answer'])
                st.write(answer)
                speak.speak(answer)