from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.loader import load_documents

# 1. Încarcă documentele
raw_docs = load_documents("docs/")

# 2. Sparge în bucăți
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.create_documents(
    [doc["content"] for doc in raw_docs],
    metadatas=[doc["metadata"] for doc in raw_docs]
)

# 3. Creează embeddings folosind DeepSeek
embeddings = OllamaEmbeddings(model="deepseek-coder:33b-instruct")
db = FAISS.from_documents(texts, embeddings)

# 4. Salvează local
db.save_local("faiss_index")
