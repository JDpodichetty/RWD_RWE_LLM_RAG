import os
import streamlit as st
import qdrant_client
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. PAGE AND MODEL CONFIGURATION ---
st.set_page_config(
    page_title="RWE Label Expansion Chatbot",
    page_icon="💊",
    layout="wide"
)

# --- Static Configurations ---
QDRANT_COLLECTION_NAME = "final_rag_collection"
MODEL_PATH = "/home/aditya/Desktop/RWE_RAG/Model/Phi-3-mini-4k-instruct-q4.gguf"
EMBEDDING_MODEL_NAME = "NeuML/bioclinical-modernbert-base-embeddings"
QDRANT_URL = "http://localhost:8001"

# --- 2. PROMPT ENGINEERING FOR MEDICAL & REGULATORY CONTEXT ---
SYSTEM_PROMPT = """
You are a specialized medical AI assistant. Your purpose is to analyze Real-World Evidence (RWE) and Real-World Data (RWD) to identify potential off-label uses for existing medicines.

When answering, you must adhere to the following strict guidelines:
1.  **Analyze the Context:** Base your answer strictly on the provided document snippets (context). Do not use any outside knowledge.
2.  **Identify Potential Uses:** Clearly state the potential off-label use suggested by the evidence.
3.  **State Limitations:** Emphasize that this is preliminary information based on RWE/RWD and is NOT a clinical recommendation.
4.  **Regulatory Disclaimer:** ALWAYS include a disclaimer stating that expanding a drug's label requires extensive clinical trials and formal approval from regulatory bodies (e.g., FDA in the US, EMA in Europe, CDSCO in India).
5.  **Advise Professional Consultation:** Conclude by strongly advising the user to consult with healthcare professionals and regulatory experts before making any decisions.

Begin your answer now based on the a context below.
---
Context: {context}
---
Question: {question}
Answer:
"""

# --- 3. CORE RAG LOGIC FUNCTIONS ---

@st.cache_resource
def load_llm():
    """Loads the quantized LLM, cached for performance."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. Please check the path.")
        st.stop()
    
    with st.spinner("Loading Large Language Model... (This may take a moment on first run)"):
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=20,
            n_batch=512,
            n_ctx=4096,
            verbose=False,
        )
    return llm

@st.cache_resource
def create_rag_chain():
    """Initializes all components and creates the RAG chain, cached for performance."""
    llm = load_llm()

    with st.spinner("Loading embeddings model and connecting to vector store..."):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        
        # --- FINAL FIX APPLIED ---
        # The content payload key has been updated to "file_path" to match
        # the key you identified in your Qdrant database.
        vector_store = Qdrant(
            client=client, 
            collection_name=QDRANT_COLLECTION_NAME, 
            embeddings=embeddings,
            content_payload_key="file_path", # <-- Correct key applied
            metadata_payload_key="metadata"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- 4. STREAMLIT UI LAYOUT ---

st.title("RWE Chatbot for Medical Label Expansion 💊")
st.markdown("This AI assistant analyzes Real-World Evidence (RWE) to identify potential off-label uses for medicines, keeping regulatory guidelines in focus.")

with st.expander("⚠️ Important Disclaimer", expanded=True):
    st.warning(
        """
        **This is an experimental AI tool and not a substitute for professional medical or legal advice.**
        - Information is for analysis purposes only, based on the provided knowledge base.
        - Off-label prescription is at the discretion of a qualified healthcare professional.
        - Expanding a drug's label requires rigorous clinical trials and formal approval from regulatory bodies.
        """
    )

# Load the RAG chain and handle potential errors
try:
    rag_chain = create_rag_chain()
    st.success("Chatbot is ready. You can now ask your questions.")
except Exception as e:
    st.error(f"Failed to initialize the chatbot. Please ensure Qdrant is running and models are correctly placed. Error: {e}")
    st.stop()

# Initialize or retrieve chat history from session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(source)

# Handle new user input
if prompt := st.chat_input("Ask about potential off-label uses based on RWE..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing evidence..."):
            result = rag_chain.invoke({"query": prompt})
            response_text = result["result"].strip()
            source_docs = [doc.page_content for doc in result["source_documents"]]
            
            st.markdown(response_text)
            with st.expander("View Sources"):
                for source in source_docs:
                    st.info(source)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": source_docs
    })

