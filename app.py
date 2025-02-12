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
# -------------------------
# CONFIGURATIONS
# -------------------------
OPENROUTER_API_KEY = st.secrets["OPEN_ROUTER_API_KEY"]
FAISS_INDEX_PATH = "faiss_index"
#embedding model
@st.cache_resource
def load_model():
    file_path = "embedding_model.pkl"
    drive_url = "https://drive.google.com/file/d/179hTGfTRQZbEczuR98J12NREDzzw8EZ7/view?usp=sharing"

    if not os.path.exists(file_path):
        gdown.download(drive_url, file_path, quiet=False)

    return file_path

EMBEDDING_MODEL_PATH = load_model()

#EMBEDDING_MODEL_PATH = "embedding_model.pkl"

# -------------------------
# LOAD FAISS INDEX & EMBEDDINGS
# -------------------------
if os.path.exists(EMBEDDING_MODEL_PATH) and os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing embedding model and FAISS index...")

    with open(EMBEDDING_MODEL_PATH, "rb") as f:
        embedding = pickle.load(f)

    vector_db = FAISS.load_local(FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True)
else:
    st.error("FAISS index or embedding model not found. Please generate them first.")
    st.stop()

# -------------------------
# SETUP FREE CHAT LLM (OPENROUTER)
# -------------------------
llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mistral-7b-instruct",
    temperature=0
)

# -------------------------
# MEMORY FOR CHAT HISTORY
# -------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------
# RETRIEVER & QA CHAIN
# -------------------------
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("💬 CampusBot")

query = st.text_input("Ask me anything:")


if query.lower() in ["hello", "hi", "namastey", "pranam"]:
    st.write("Hello, I hope you are doing well 😊.")
elif query.lower() in ["who are you?", "who are you", "what's your name?"]:
    st.write("I am CampusBot from IIIT RANCHI, created by Devam Singh from batch 2022-2026, BTech CSE (Specialization in DSAI).")



else:
    try:
        # ✅ Using `.invoke()` to call the chain
        result = qa_chain.invoke({"question": query})
        st.write(result["answer"])
    except Exception as e:
        st.error(f"Error: {e}")
