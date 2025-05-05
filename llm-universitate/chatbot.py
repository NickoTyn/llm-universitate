from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import gradio as gr

# Încarcă baza de date FAISS cu embeddings generate de DeepSeek
db = FAISS.load_local(
    "faiss_index",
    OllamaEmbeddings(model="deepseek-coder:6.7b-instruct"),  # sau "33b-instruct"
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever()

# Inițializează LLM DeepSeek local
llm = Ollama(model="deepseek-coder:6.7b-instruct")  # sau "33b-instruct"

# Configurează lanțul de QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Interfața de chat
def chatbot_interface(question, history=[]):
    prompt = f"Ești un asistent universitar politicos. Răspunde clar, în limba română:\n{question}"
    answer = qa.run(prompt)
    return answer

# Pornește interfața Gradio în browser
gr.ChatInterface(fn=chatbot_interface).launch(inbrowser=True)
