import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from huggingface_hub import login

# Title and description
st.title("RAG-based Chatbot")
st.write("This app uses a Retrieval-Augmented Generation (RAG) pipeline to answer queries based on a knowledge base.")

# Hugging Face API login
api_key = st.text_input("Enter your Hugging Face API key:", type="password")
if api_key:
    try:
        login(token=api_key)
        st.success("Successfully logged in to Hugging Face!")
    except Exception as e:
        st.error(f"Login failed: {e}")

# Upload knowledge base file
uploaded_file = st.file_uploader("Upload a knowledge base file (e.g., .txt):", type=["txt"])
if uploaded_file is not None and api_key:
    # Load the knowledge base
    loader = TextLoader(uploaded_file)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Initialize embedding model and vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    st.success("Knowledge base successfully processed!")

    # Set up LLaMA model
    llm = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B",
        device_map="auto",
        torch_dtype="auto"
    )

    # Function for RAG chatbot
    def rag_chatbot(query):
        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate response using LLaMA
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        response = llm(prompt, max_length=300, num_return_sequences=1)
        return response[0]["generated_text"]

    # User query input
    query = st.text_input("Enter your query:")
    if query:
        # Generate and display the response
        with st.spinner("Generating response..."):
            response = rag_chatbot(query)
        st.write("**Response:**")
        st.write(response)

elif not api_key:
    st.warning("Please provide your Hugging Face API key to proceed.")
else:
    st.warning("Please upload a knowledge base file to proceed.")
