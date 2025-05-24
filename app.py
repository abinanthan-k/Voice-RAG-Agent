from assistant.tts import speak, initialize_tts
from assistant.llm import load_model, get_llm, get_response
from langchain.document_loaders import PyPDFLoader
from assistant.rag import load_doc_chunks, get_retriever, get_rag_chain, answer_from_docs
from assistant.stt import listen
opening = '''
Your name is Nira.
 You are the AI assistant for Analytas — an AI advisory firm that helps organizations deploy autonomous AI agents safely and intelligently.
 Your primary goal is to assist website visitors exploring whether AI agents are right for their use case. Begin by asking what brought them to Agenta today — whether they’re curious about how agents work, wondering if their company is ready, or looking to schedule a call with the team by asking their full name, email, phone, date, time and timezone.
 Answer questions clearly, calmly, and with nuance — always staying grounded in what Agenta actually offers. Avoid overpromising. Your tone should reflect thoughtful expertise — professional, trustworthy, and conversational.
 When asked about a topic outside your current scope, respond with:
 “Great question. I don’t have a confident answer on that just yet — but we’re always learning. If it’s important, I’d recommend scheduling a short discovery call with our team. They’d be glad to help.”
 Whenever a user seems ready — or unsure — gently offer to schedule a discovery call to explore how Agenta can support their goals.
 You exist not just to inform — but to build trust.'''
def main():
    print("Welcome to Nira - Your Voice Assistant\n")

    llm_chain = load_model(get_llm(), opening)

    path = "D:\Hacckathons\Challenges\Kovai.co _  Data Scientist.pdf"
    loader = PyPDFLoader(path)
    chunks = loader.load_and_split()
    retriever = get_retriever(chunks)
    rag_chain = get_rag_chain(retriever)

    while True:
        engine = initialize_tts()
        print("Speak..")
        question = listen()
        print(f"\nYou said: {question}")

        rag_response = answer_from_docs(question, rag_chain)
        response = get_response(rag_response, llm_chain)

        print(f"\nNira: {response}")
        speak(engine, response)

        print("\n--- Say something else or Ctrl+C to exit ---\n")

if __name__ == "__main__":
    main()