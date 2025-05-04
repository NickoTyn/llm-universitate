
from pathlib import Path
import fitz  # PyMuPDF
import docx

def load_documents(directory):
    documents = []
    for file in Path(directory).glob("*"):
        text = ""
        if file.suffix == ".pdf":
            doc = fitz.open(file)
            for page in doc:
                text += page.get_text()
        elif file.suffix in [".docx", ".DOCX"]:
            doc = docx.Document(file)
            text += "\n".join([para.text for para in doc.paragraphs])
        else:
            continue
        documents.append({"content": text, "metadata": {"source": str(file.name)}})
    return documents
