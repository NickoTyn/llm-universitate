
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import gradio as gr

# Încarcă DB
db = FAISS.load_local("faiss_index", OllamaEmbeddings(model="mistral"), allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# LLM local
llm = Ollama(model="mistral")

# Chain de QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Interfața chat
def chatbot_interface(question, history=[]):
    prompt = f"Răspunde în limba română la următoarea întrebare:\n{question}"
    answer = qa.run(prompt)
    return answer

gr.ChatInterface(fn=chatbot_interface).launch(inbrowser=True)
