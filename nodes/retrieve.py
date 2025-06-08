from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

def load_prompt():
    system_prompt = (
        '''
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you dont know the answer, say that "I don't know. It is not provided in the context" 
        Use three sentence as maximum and keep the answer concise.\n\n
        {context}
        '''
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt

def load_rag_chain(llm, retriever):
    if retriever is None:
        raise ValueError("Retriever cannot be None")    
    prompt = load_prompt()
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain