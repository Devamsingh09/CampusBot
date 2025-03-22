import os
import pickle
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
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

    # Verify if the downloaded file is valid
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

# SYSTEM PROMPT TEMPLATE (Fixed)
SYSTEM_PROMPT_TEMPLATE = """
You are CampusBot, an AI assistant providing fact-based responses using retrieved knowledge from a FAISS index. 
Always ground your answers in retrieved content and avoid speculation. 
If the retrieved data is insufficient, state so rather than guessing. 
Provide well-structured, concise, and informative answers. Cite sources when relevant. 
Maintain clarity and a professional yet engaging tone.

Context: {context}

Question: {question}

Helpful Answer:
"""

# Convert system prompt to a valid PromptTemplate
prompt_template = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_PROMPT_TEMPLATE)

# SETUP FREE CHAT LLM (OPENROUTER)
llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mistral-7b-instruct",
    temperature=0,
)

# MEMORY FOR CHAT HISTORY
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# RETRIEVER & QA CHAIN
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain=load_qa_chain(
        llm=llm,
        chain_type="stuff",  # "stuff" is good for short answers
        prompt=prompt_template  # ✅ Fixed: Using PromptTemplate instead of raw string
    )
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
            st.error(f"Error processing query: {e}")
else:
    st.write("Ask me anything!")
