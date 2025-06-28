#rag_app.py <-- web-based version of the interactive program

import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st # Import Streamlit

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Load Environment Variables ---
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPENAI_API_KEY") # Use .get() for safer access

if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop the Streamlit app if the key is missing

# --- 2. Streamlit UI Setup ---
#st.set_page_config(page_title="Philippine History RAG Chatbot", layout="centered")
st.set_page_config(page_title="Philippine History Chatbot", layout="centered")
st.title("Philippine History Chatbot")
st.markdown("Ask me anything about Philippine history based on the document provided.")

# --- 3. RAG System Initialization (Cached for performance) ---
# Use Streamlit's caching to avoid re-loading and re-processing the document on every rerun
@st.cache_resource
def setup_rag_system():
    # print("Loading document and setting up vector database...") # This won't show in Streamlit console
    try:
        loaded_document = TextLoader("./data/PhilHistory.txt", encoding='utf-8').load()
    except Exception as e:
        st.error(f"Error loading document: {e}. Make sure 'PhilHistory.txt' is in the 'data' folder.")
        st.stop()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks_of_text = text_splitter.split_documents(loaded_document)

    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks_of_text, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k":3})

    # LangChain Expression Language (LCEL) Chain Setup
    template = """ Answer the question based only on the ff: context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0.0) # Added temperature for more consistent answers

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    # print("Vector database setup complete.\n")
    return chain

# Initialize the RAG chain
with st.spinner("Setting up the RAG system... This might take a moment."):
    rag_chain = setup_rag_system()

# --- 4. User Input and Response Display ---
# Create a text input for the user's question
user_question = st.text_input("Your Question:", placeholder="e.g., Who was Jose Rizal?")

# Create a button to trigger the response
if st.button("Get Answer"):
    if user_question:
        with st.spinner("Searching for an answer..."):
            try:
                response = rag_chain.invoke(user_question)
                st.info("AI Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while getting the answer: {e}")
                st.error("Please check your API key, document, or internet connection.")
    else:
        st.warning("Please type a question to get an answer.")

st.markdown("---")
st.caption("Powered by LangChain, OpenAI, and Streamlit")