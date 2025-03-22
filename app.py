import os
import pickle
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import gdown

load_dotenv()

# CONFIGURATIONS
OPENROUTER_API_KEY = st.secrets["OPEN_ROUTER_API_KEY"]
FAISS_INDEX_PATH = "faiss_index"

# Embedding model loader
@st.cache_resource
def load_model():
    file_path = "embedding_model.pkl"
    file_id = "179hTGfTRQZbEczuR98J12NREDzzw8EZ7"  
    drive_url = f"https://drive.google.com/uc?id={file_id}"  

    if not os.path.exists(file_path):
        print("Downloading embedding model from Google Drive...")
        gdown.download(drive_url, file_path, quiet=False)

    # Verifying if the downloaded file is valid
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print("Download successful!")
    else:
        raise ValueError("Downloaded file is invalid or empty. Please check the Google Drive link.")

    return file_path

EMBEDDING_MODEL_PATH = load_model()

# LOADING FAISS INDEX & EMBEDDINGS
if os.path.exists(EMBEDDING_MODEL_PATH) and os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing embedding model and FAISS index...")
    with open(EMBEDDING_MODEL_PATH, "rb") as f:
        embedding = pickle.load(f)
    
    vector_db = FAISS.load_local(FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True)
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
        "You are CampusBot, an AI assistant designed to provide fact-based and contextually relevant answers using retrieved documents. "
        "Always ensure your responses align with the retrieved content, avoiding speculation. "
        "If the retrieved data is insufficient, acknowledge the limitation rather than guessing. "
        "Maintain a formal yet approachable tone, and summarize key points effectively. "
        "Provide additional clarifications where necessary, and cite sources when applicable."
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
    if query.lower() in ["hello", "hi", "namastey", "pranam"]:
        st.write("Hello, I hope you are doing well 😊.")
    elif query.lower() in ["who are you?", "who are you", "what's your name?", "who created you?"]:
        st.write("I am CampusBot from IIIT RANCHI, created by Devam Singh from batch 2022-2026, BTech CSE (Specialization in DSAI).")
    else:
        try:
            result = qa_chain.invoke({"question": query})
            st.write(result["answer"])
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.write("Ask me anything!")
