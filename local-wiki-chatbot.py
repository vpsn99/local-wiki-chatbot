"""
local-wiki-chatbot.py
--------------------
A hybrid chatbot that can answer from local documents (PDF, DOCX, TXT)
or from Wikipedia. 
Uses 1) HuggingFace embeddings, 2) ChromaDB for local vector
search, and 3) Gradio for an interactive user interface.

Author: Virendra Pratap Singh
License: MIT
"""

# =====================================================
# 1. Imports
# =====================================================
import os
import re
import shutil
import docx2txt
import PyPDF2
import wikipediaapi
import gradio as gr
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# =====================================================
# 2. Configuration
# =====================================================
DATA_PATH = "datafiles"
PERSIST_DIR = "chroma_db"
LOCAL_SIMILARITY_THRESHOLD = 0.35
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Wikipedia API client with a custom user agent (required)
wiki_api = wikipediaapi.Wikipedia(
    language="en",
    user_agent="LocalWikiChatbot/1.0 (+https://github.com/vpsn99/local-wiki-chatbot)"
)

# =====================================================
# 3. File Reading and Chunking
# =====================================================
def read_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text


def read_docx(file_path):
    """Extract text from a DOCX file."""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def read_txt(file_path):
    """Extract text from a TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

# =====================================================
# 4. Build or Rebuild the Local Vector Database
# =====================================================
def build_chroma_from_folder(folder, persist_dir, embedding_function):
    """Read supported documents and create a Chroma vector store."""
    documents = []
    print(f"Indexing local files from: {os.path.abspath(folder)}")

    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        if not os.path.isfile(path):
            continue

        ext = fn.lower().split(".")[-1]
        text = ""
        if ext == "pdf":
            text = read_pdf(path)
        elif ext in ("docx", "doc"):
            text = read_docx(path)
        elif ext == "txt":
            text = read_txt(path)
        else:
            continue

        if not text.strip():
            continue

        for chunk in chunk_text(text):
            documents.append(Document(page_content=chunk, metadata={"source": fn}))

    if not documents:
        print("No readable files found.")
        return None

    # Rebuild persistence directory
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"Indexed {len(documents)} text chunks.")
    return vectorstore

# =====================================================
# 5. Local and Wikipedia Query Functions
# =====================================================
def query_local(vectorstore, question, k=3):
    """Search the local Chroma database for a relevant answer."""
    if not vectorstore:
        return None, "No local data is indexed yet."

    results = vectorstore.similarity_search_with_score(question, k=k)
    if not results:
        return None, "No relevant local documents found."

    best_doc, score = results[0]
    if score < LOCAL_SIMILARITY_THRESHOLD:
        return None, f"Local match found but below relevance threshold (score={score:.2f})."
    return best_doc, f"Answer from {best_doc.metadata['source']} (similarity={score:.2f})"


def query_wiki(question):
    """Search Wikipedia for the given question."""
    try:
        query_term = re.sub(r"[^a-zA-Z0-9\s]", "", question).strip().title()
        page = wiki_api.page(query_term)
        if not page.exists():
            return f"No Wikipedia page found for '{query_term}'."
        summary = page.summary[:800] + "..."
        return f"Source: Wikipedia\nTitle: {page.title}\n\n{summary}\n\nLink: {page.fullurl}"
    except Exception as e:
        return f"Wikipedia query failed: {e}"

# =====================================================
# 6. Combined Response Logic
# =====================================================
def get_response(query, mode, vectorstore):
    """Decide whether to use local data or Wikipedia for a response."""
    query = query.strip()
    if not query:
        return "Please enter a question."

    if mode == "local":
        doc, info = query_local(vectorstore, query)
        if doc:
            return f"{info}\n\n{doc.page_content}"
        else:
            wiki_answer = query_wiki(query)
            return f"{info}\n\n{wiki_answer}"

    elif mode == "wiki":
        return query_wiki(query)

    else:
        return "Invalid mode selection."

# =====================================================
# 7. Gradio Interface
# =====================================================
def launch_app(vectorstore):
    """Launch the Gradio web interface."""
    with gr.Blocks(title="Local and Wikipedia Chatbot") as demo:
        gr.Markdown("# Local and Wikipedia Chatbot")
        gr.Markdown("Ask a question and choose whether to search locally or on Wikipedia.")

        with gr.Row():
            query_box = gr.Textbox(label="Your question", placeholder="e.g., What is Canada?")
            mode_box = gr.Radio(["local", "wiki"], value="local", label="Search Mode")

        output = gr.Markdown()

        def chat_interface(query, mode):
            return get_response(query, mode, vectorstore)

        query_box.submit(chat_interface, [query_box, mode_box], [output])
        mode_box.change(chat_interface, [query_box, mode_box], [output])

    demo.launch(debug=False, share=False)

# =====================================================
# 8. Script Entry Point
# =====================================================
if __name__ == "__main__":
    print("Initializing embedding model...")
    embedding_function = SentenceTransformer(EMBED_MODEL_NAME)

    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
        vectorstore = build_chroma_from_folder(DATA_PATH, PERSIST_DIR, embedding_function)
    else:
        print("Using existing ChromaDB index.")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_function)

    print("Starting chatbot...")
    launch_app(vectorstore)
