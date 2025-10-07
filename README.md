# local-wiki-chatbot

A lightweight chatbot that answers questions from your **local documents** (PDF, DOCX, TXT) and/or **Wikipedia**.  
It uses HuggingFace sentence embeddings for semantic search, ChromaDB for vector storage, and Gradio for a simple web UI.

---

## Features

- Local document search with semantic similarity (no keywords required)
- Wikipedia integration for general knowledge
- Two modes:
  - **Local**: search local files first; if nothing relevant is found, fall back to Wikipedia
  - **Wiki**: query only Wikipedia
- Persistent local vector database with ChromaDB
- Works fully offline in Local mode (indexing + answering)
- Minimal UI via Gradio

---

## Requirements

- Python 3.10 or 3.11
- OS: Windows, macOS, or Linux
- No API keys required

Install project dependencies with:

```bash
pip install -r requirements.txt
```

> Versions are pinned in `requirements.txt` for reproducibility.

---

## Project structure

```
local-wiki-chatbot/
│
├── local_wiki_chatbot.py       # Main application script
├── local_wiki_chatbot.ipynb    # Optional notebook (if you include it)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── datafiles/                  # Place your local PDFs/DOCX/TXT here
│   └── .gitkeep
│
└── chroma_db/                  # Auto-generated ChromaDB persistence
    └── README.md               # Optional: explain that this is generated
```

> Do not commit real documents or the `chroma_db/` folder. Both are ignored via `.gitignore`.

---

## Setup

1) **Create and activate a virtual environment**

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux (bash/zsh):**
```bash
python -m venv .venv
source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

3) **Add documents (optional)**
- Put `.pdf`, `.docx`, or `.txt` files into the `datafiles/` directory.
- The app will index them on first run.

---

## Run

```bash
python local_wiki_chatbot.py
```

- A local Gradio link will appear in the terminal. Open it in your browser.
- Select a mode:
  - **Local**: searches your documents first; if nothing relevant is found, Wikipedia is used as a fallback.
  - **Wiki**: queries only Wikipedia.

---

## Configuration

Edit these constants near the top of `local_wiki_chatbot.py` as needed:

| Constant | Purpose | Default |
|---|---|---|
| `DATA_PATH` | Directory containing local documents | `"datafiles"` |
| `PERSIST_DIR` | Directory for ChromaDB persistence | `"chroma_db"` |
| `LOCAL_SIMILARITY_THRESHOLD` | Minimum similarity to accept a local match | `0.35` |
| `EMBED_MODEL_NAME` | Sentence-transformers embedding model | `"all-MiniLM-L6-v2"` |

**Wikipedia client:**  
The script uses `wikipedia-api` with a custom user agent, per Wikimedia policy. Update the `user_agent` string to point to your GitHub repo.

---

## How it works

1) **Indexing**  
   - Each document is parsed (`PyPDF2`, `docx2txt`, or plain text).
   - Text is split into overlapping chunks (default: 500 chars with 50 overlap).
   - Chunks are embedded with a HuggingFace model (default: `all-MiniLM-L6-v2`) and stored in ChromaDB.

2) **Querying**  
   - In **Local** mode, the question is embedded and compared (cosine similarity) against local chunks.
   - If the best local match is below `LOCAL_SIMILARITY_THRESHOLD`, the answer falls back to Wikipedia.
   - In **Wiki** mode, the app queries Wikipedia directly and returns the page summary and link.

3) **UI**  
   - Gradio provides a one-page web interface with a textbox and a radio button for mode selection.

---

## Rebuilding the index

The app builds the index automatically on first run.  
If you add or change files and want to force a full rebuild:

```bash
# stop the app if it's running
# then remove the persistent DB
rm -rf chroma_db          # macOS/Linux
rmdir /s /q chroma_db     # Windows (Command Prompt)
# or: powershell: Remove-Item chroma_db -Recurse -Force
```

Then re-run:
```bash
python local_wiki_chatbot.py
```

---

## Troubleshooting

**“No relevant local documents found.”**  
- Ensure files are in `datafiles/` and readable.
- Lower `LOCAL_SIMILARITY_THRESHOLD` slightly (e.g., 0.30).
- Include clearer, definition-style sentences in documents (e.g., “Canada is a country in North America …”).

**“Database locked” on Windows when rebuilding**  
- Close any running instance of the app (and other Python sessions).
- Delete the `chroma_db/` folder, then run again.

**Wikipedia returns the wrong page**  
- Rephrase the question to include the exact entity name (e.g., “Canada country”).
- The script normalizes queries, but ambiguous terms may still resolve to popular pages.

**Dependency issues**  
- Use the pinned versions in `requirements.txt`.
- If you use a different Python version, re-create a fresh virtual environment.

---

## Extensibility ideas

- Add other sources (CSV, Markdown, Google Drive, Notion)
- Re-rank top-k passages with a cross-encoder for better precision
- Summarize multi-chunk answers with a local or hosted LLM
- Dockerize for deployment
- Add basic telemetry (queries per minute, index size)
- Support multi-language Wikipedia

---

## Acknowledgements

- [Sentence-Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [LangChain (Community)](https://python.langchain.com/)
- [Gradio](https://gradio.app/)
- [wikipedia-api](https://pypi.org/project/Wikipedia-API/)

---

## License

MIT License © 2025 Virendra Pratap Singh - vpsn_99@yahoo.com  
See `LICENSE` for details.
