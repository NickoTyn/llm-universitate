from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import re

# Inițializare LLM și vector DB cu DeepSeek
embeddings = OllamaEmbeddings(model="deepseek-coder:33b-instruct")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
llm = Ollama(model="deepseek-coder:33b-instruct")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

app = FastAPI()

# Memorie conversațională simplă per IP
conversation_history = {}

class QueryRequest(BaseModel):
    question: str

def is_english(text: str) -> bool:
    common_words = r'\b(?:the|is|and|you|this|that|for|with|are|have|be|from|on|as|at|by|it|an|if|do|not|can|will|or)\b'
    english_words = re.findall(common_words, text.lower())
    return len(english_words) >= 5

@app.post("/ask")
async def ask_question(data: QueryRequest, request: Request):
    user_input = data.question.strip()
    user_ip = request.client.host

    if user_ip not in conversation_history:
        conversation_history[user_ip] = []

    if user_input.lower() in ["salut", "buna", "bună", "hello", "hi"]:
        return {
            "answer": "Salut! Sunt asistentul universității. Cu ce te pot ajuta legat de burse, admitere sau alte informații?"
        }

    # Construiește contextul din ultimele 2-3 întrebări și răspunsuri
    history = conversation_history[user_ip][-3:]
    context = "\n".join([f"Utilizator: {q}\nAsistent: {a}" for q, a in history])

    system_prompt = (
        "Ești un asistent universitar inteligent și politicos. Răspunde întotdeauna doar în limba română. "
        "Nu folosi engleza. Fii foarte clar, foarte scurt și răspunde doar la ceea ce s-a întrebat. "
        "Nu repeta informații deja menționate. "
        "Dacă întrebarea nu este clară, cere clarificări."
    )

    full_prompt = f"{system_prompt}{context}\nUtilizator: {user_input}\nAsistent:"

    answer = qa.run(full_prompt)

    if not answer or len(answer.strip()) < 20:
        return {"answer": "Îmi pare rău, nu am găsit suficiente informații pentru a răspunde. Poți reformula întrebarea?"}

    if is_english(answer):
        retry_prompt = (
            f"{system_prompt}{context}\nATENȚIE: Răspunsul precedent a fost în engleză. Refă răspunsul în română.\n"
            f"Utilizator: {user_input}\nAsistent:"
        )
        answer = qa.run(retry_prompt)

    # Stochează întrebarea și răspunsul curent
    conversation_history[user_ip].append((user_input, answer))
    if len(conversation_history[user_ip]) > 5:
        conversation_history[user_ip] = conversation_history[user_ip][-5:]

    return {"answer": answer}
