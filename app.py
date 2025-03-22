import os
import pickle
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import gdown

# Load environment variables
load_dotenv()

# CONFIGURATIONS
OPENROUTER_API_KEY = st.secrets["OPEN_ROUTER_API_KEY"] if "OPEN_ROUTER_API_KEY" in st.secrets else os.getenv("OPEN_ROUTER_API_KEY")
FAISS_INDEX_PATH = "faiss_index"

# Embedding model loader
@st.cache_resource
def load_model():
    file_path = "embedding_model.pkl"
    file_id = "179hTGfTRQZbEczuR98J12NREDzzw8EZ7"  
    drive_url = f"https://drive.google.com/uc?id={file_id}"  

    if not os.path.exists(file_path):
        st.info("Downloading embedding model from Google Drive...")
        gdown.download(drive_url, file_path, quiet=False)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_path
    else:
        st.error("Downloaded file is invalid or empty. Please check the Google Drive link.")
        st.stop()

EMBEDDING_MODEL_PATH = load_model()

# LOADING FAISS INDEX & EMBEDDINGS
if os.path.exists(EMBEDDING_MODEL_PATH) and os.path.exists(FAISS_INDEX_PATH):
    st.success("Loading existing embedding model and FAISS index...")
    with open(EMBEDDING_MODEL_PATH, "rb") as f:
        embedding = pickle.load(f)
    
    try:
        vector_db = FAISS.load_local(FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.stop()
else:
    st.error("FAISS index or embedding model not found. Please generate them first.")
    st.stop()

# SETUP FREE CHAT LLM (OPENROUTER)
llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mistral-7b-instruct",
    temperature=0,
    system_message=(
        "You are CampusBot, an AI assistant providing fact-based responses using retrieved knowledge from a FAISS index. "
        "Always ground your answers in retrieved content and avoid speculation. If the retrieved data is insufficient, state so rather than guessing. "
        "Provide well-structured, concise, and informative answers. Cite sources when relevant. Maintain clarity and a professional yet engaging tone."
    )
)

# MEMORY FOR CHAT HISTORY
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# RETRIEVER & QA CHAIN
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# STREAMLIT UI
st.title("💬 CampusBot")
query = st.text_input("Ask me anything!")

if query:
    try:
        if query.lower() in ["hello", "hi", "namastey", "pranam"]:
            st.write("Hello, I hope you are doing well 😊.")
        elif query.lower() in ["who are you?", "who are you", "what's your name?", "who created you?"]:
            st.write("I am CampusBot from IIIT RANCHI, created by Devam Singh from batch 2022-2026, BTech CSE (Specialization in DSAI).")
        else:
            result = qa_chain.invoke({"question": query})
            st.write(result["answer"])
    except Exception as e:
        st.error(f"Error processing query: {e}")
else:
    st.write("Ask me anything!")
