# 🎬 YouTube Transcript Analyzer

A RAG (Retrieval-Augmented Generation) powered application that fetches transcripts from YouTube videos, indexes them into a vector store, and lets you ask natural language questions about the video's content — all through a clean Streamlit chat interface.

---

## 📌 What It Does

Paste any YouTube URL, and this app will:
1. Fetch the video's English transcript automatically
2. Split it into chunks and embed them using a local HuggingFace model
3. Store the embeddings in a FAISS vector store
4. Let you ask any question about the video in a conversational chat UI
5. Use `Mistral-7B-Instruct` via HuggingFace Inference API to generate accurate, context-aware answers

---

## 🗂️ Project Structure

```
YouTube-Transcript-Analyzer/
│
├── app.py              # Manual RAG pipeline (step-by-step, no chain)
├── app_chain.py        # Same pipeline refactored using LangChain LCEL chains
├── streamlit_app.py    # Full interactive Streamlit web app
├── requirements.txt    # All Python dependencies
└── .gitignore
```

### File Breakdown

| File | Purpose |
|---|---|
| `app.py` | A script-style RAG pipeline. Hardcodes a `video_id`, fetches the transcript, chunks it, embeds it with FAISS, retrieves relevant chunks manually, and queries Mistral-7B. Good for understanding the flow step by step. |
| `app_chain.py` | A cleaner, chained version of `app.py` using LangChain's `RunnableParallel`, `RunnableLambda`, and `RunnablePassthrough` to compose the full RAG pipeline as a single chain. |
| `streamlit_app.py` | The production-ready web app. Accepts any YouTube URL via sidebar, builds the vector store on the fly, and provides a persistent chat interface to query the video. |

---

## ⚙️ How It Works — RAG Pipeline

The application follows the standard **RAG (Retrieval-Augmented Generation)** pattern across four stages:

### Step 1 — Indexing
- **Transcript Fetching**: Uses `youtube-transcript-api` to fetch the English transcript for a given `video_id`.
- **Text Splitting**: The raw transcript is chunked using `RecursiveCharacterTextSplitter` (chunk size: 1000–1500 tokens, overlap: 200–300 tokens) to preserve context across chunk boundaries.
- **Embedding**: Chunks are embedded using the `all-mpnet-base-v2` sentence transformer model from HuggingFace.
- **Vector Store**: The embedded chunks are stored in a **FAISS** (Facebook AI Similarity Search) in-memory vector store for fast similarity lookup.

### Step 2 — Retrieval
- A FAISS retriever performs a similarity search over the stored chunks, returning the top `k` most relevant segments for a given question (k=4 in scripts, k=8 in the Streamlit app).

### Step 3 — Augmentation
- Retrieved chunks are formatted into a single context string.
- A `PromptTemplate` injects both the `context` and the `question` into a structured prompt, instructing the model to answer only from the transcript context.

### Step 4 — Generation
- The prompt is sent to **Mistral-7B-Instruct-v0.2** hosted on the HuggingFace Inference API via `HuggingFaceEndpoint`.
- The model generates a context-grounded answer, which is parsed to a clean string and returned to the user.

---

## 🖥️ Streamlit App Features

- **Sidebar URL input** — paste any standard YouTube or `youtu.be` URL
- **Automatic video ID extraction** via regex
- **Session state persistence** — the vector store and chat history survive re-runs within a session
- **Cached models** — both the embeddings model and the LLM are loaded once per session using `@st.cache_resource`
- **Chat UI** — uses `st.chat_input` and `st.chat_message` for a modern conversational experience
- **Graceful error handling** — surfaces errors for missing URLs, unavailable transcripts, or un-indexed videos

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MARKEE-code-sketch/YouTube-Transcript-Analyzer.git
cd YouTube-Transcript-Analyzer
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

You can get a free API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 5. Run the app

```bash
streamlit run streamlit_app.py
```

Or run the scripts directly:

```bash
python app.py         # Basic RAG pipeline
python app_chain.py   # Chained RAG pipeline
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-core`, `langchain-community` | RAG pipeline orchestration |
| `langchain-huggingface` | HuggingFace LLM and embeddings integration |
| `youtube-transcript-api` | Fetching YouTube video transcripts |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Local embedding model (`all-mpnet-base-v2`) |
| `transformers`, `huggingface-hub` | Model loading and inference |
| `streamlit` | Web application UI |
| `python-dotenv` | Environment variable management |
| `torch`, `numpy`, `scikit-learn` | ML utilities |
| `tiktoken` | Token counting for text splitting |

---

## 🔑 Requirements

- Python 3.9+
- A [HuggingFace account](https://huggingface.co/) with an API token (free tier works)
- The target YouTube video must have English captions/transcripts enabled

---

## 💡 Example Usage

1. Launch the Streamlit app
2. Paste a YouTube URL in the sidebar (e.g., `https://www.youtube.com/watch?v=PnqJllk3RfA`)
3. Click **Analyze Video** and wait for indexing to complete
4. Type a question in the chat box:
   - *"Can you summarize this video?"*
   - *"What are the main topics discussed?"*
   - *"Was happiness mentioned in this podcast?"*

---

## 🛠️ Tech Stack

- **Python** — core language
- **LangChain** — RAG pipeline and LCEL chain composition
- **HuggingFace** — embeddings (`all-mpnet-base-v2`) + LLM (`Mistral-7B-Instruct-v0.2`)
- **FAISS** — vector similarity search
- **Streamlit** — interactive web UI
- **youtube-transcript-api** — transcript retrieval

---

## 📄 License

This project is open source. Feel free to fork, extend, and build on top of it.
