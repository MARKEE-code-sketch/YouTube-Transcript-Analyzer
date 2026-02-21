import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="YouTube Transcript Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------- HEADER ----------
st.title("🎬 YouTube Transcript Analyzer")
st.caption("Ask questions from any YouTube video using AI")

# ---------- SESSION STATE ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- FUNCTIONS ----------
def extract_video_id(url):
    pattern = r"(?:v=|youtu\.be/)([^&\n?#]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def fetch_transcript(video_id):
    yt_api = YouTubeTranscriptApi()
    try:
        transcript_list = yt_api.fetch(video_id=video_id, languages=["en"])
        transcript = ""
        for segment in transcript_list:
            transcript += segment.text + " "
        return transcript
    except TranscriptsDisabled:
        return None


@st.cache_resource
def get_embeddings_model():
    """Cache the embeddings model to avoid reloading"""
    return HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


def build_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


@st.cache_resource
def get_llm_model():
    """Cache the LLM model to avoid reloading"""
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)


def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY using transcript context.
If context is insufficient, say you don't know.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    model = get_llm_model()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    return parallel_chain | prompt | model | parser


# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Video Setup")

video_url = st.sidebar.text_input("Paste YouTube URL")

if st.sidebar.button("Analyze Video"):
    if not video_url:
        st.sidebar.error("Enter a YouTube URL")
    else:
        with st.spinner("Fetching transcript..."):
            video_id = extract_video_id(video_url)
            transcript = fetch_transcript(video_id)

        if transcript is None:
            st.error("No transcript available.")
        else:
            with st.spinner("Creating embeddings..."):
                st.session_state.vector_store = build_vector_store(transcript)
            st.success("Video indexed successfully!")


# ---------- CHAT UI ----------
st.subheader("💬 Ask Questions")

user_question = st.chat_input("Ask something about the video...")

if user_question and st.session_state.vector_store:
    st.session_state.chat_history.append(("user", user_question))

    with st.spinner("Thinking..."):
        rag_chain = create_rag_chain(st.session_state.vector_store)
        answer = rag_chain.invoke(user_question)

    st.session_state.chat_history.append(("assistant", answer))


# ---------- DISPLAY CHAT ----------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)


# ---------- EMPTY STATE ----------
if not st.session_state.vector_store:
    st.info("👈 Add a YouTube link in sidebar and click Analyze Video")
