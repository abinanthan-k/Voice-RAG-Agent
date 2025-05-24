from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

USE_GEMINI = True

def get_llm():
    if USE_GEMINI:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyCXz5o_ZTsrjfRzKdJrrIT_4tQGNcBqkz4")
        return llm
    else:
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key="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")
        return llm

def load_model(llm, opening):
    system_template = "{opening}"
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
    )
    return LLMChain(llm=llm, prompt=prompt_template)

def get_response(rag_response, llm_chain):
    result = llm_chain.run(
        text=rag_response
    )
    return result.content




