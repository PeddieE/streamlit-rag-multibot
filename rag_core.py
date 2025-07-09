#version for re-deployment

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
import io

# --- START OF MODIFICATION ---

# Get the base path for ChromaDB persistence from an environment variable.
# On Render, we will set CHROMA_PERSIST_BASE_DIR to your disk's mount path (e.g., "/data").
# Locally, it will default to '.' (the current directory) for development.
CHROMA_PERSIST_BASE_DIR = os.getenv("CHROMA_PERSIST_BASE_DIR", ".")

# Define the directory where ChromaDB collections will be stored within the persistent base path
# This will create something like:
#   - './chroma_db_data' locally
#   - '/data/chroma_db_data' on Render (assuming /data is your mount path)
CHROMA_DB_STORAGE_FOLDER = os.path.join(CHROMA_PERSIST_BASE_DIR, "chroma_db_data")

# Ensure the directory exists (important for both local and Render environments)
os.makedirs(CHROMA_DB_STORAGE_FOLDER, exist_ok=True)

print(f"DEBUG: ChromaDB will persist data in: {CHROMA_DB_STORAGE_FOLDER}") # Very useful for Render logs!

# --- END OF MODIFICATION ---


# Function to load a PDF and split it into documents
def load_and_split_pdf(uploaded_file):
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to set up the RAG pipeline
def setup_rag_pipeline(pdf_name: str):
    # Ensure OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API Key not found. Please set it in your Render environment variables.")
        return None, None

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # --- START OF MODIFICATION ---
    # Define the specific collection path for this PDF within the main storage folder
    # Example: /data/chroma_db_data/my_document_name_db
    collection_persist_path = os.path.join(CHROMA_DB_STORAGE_FOLDER, f"{pdf_name.replace('.', '_')}_db")
    print(f"DEBUG: ChromaDB collection for '{pdf_name}' will be at: {collection_persist_path}") # Debug print

    # Initialize Chroma vectorstore for persistence
    # Use the collection_persist_path here
    vectorstore = Chroma(
        persist_directory=collection_persist_path,
        embedding_function=embeddings
    )
    # --- END OF MODIFICATION ---

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain, vectorstore

# Streamlit UI
st.title("📄 PDF-based RAG Chatbot")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "current_pdf_name" not in st.session_state:
    st.session_state["current_pdf_name"] = None

# Process uploaded PDF
if uploaded_file is not None and st.session_state["current_pdf_name"] != uploaded_file.name:
    st.info("Processing PDF... This might take a moment depending on the document size.")
    
    # Save the uploaded file to a temporary location for PyPDFLoader
    # Render's ephemeral filesystem means this is fine for initial processing,
    # but the ChromaDB itself must go to the persistent disk.
    with open(os.path.join("./temp_pdf_uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Ensure temp_pdf_uploads directory exists for the PyPDFLoader
    os.makedirs("./temp_pdf_uploads", exist_ok=True)
    temp_pdf_path = os.path.join("./temp_pdf_uploads", uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    texts = load_and_split_pdf(temp_pdf_path)
    
    # Initialize RAG pipeline (this is where ChromaDB is created/loaded)
    qa_chain, vectorstore = setup_rag_pipeline(uploaded_file.name)
    
    if qa_chain and vectorstore:
        # Add texts to vectorstore (will persist due to persist_directory)
        vectorstore.add_documents(texts)
        st.session_state["qa_chain"] = qa_chain
        st.session_state["vectorstore"] = vectorstore
        st.session_state["current_pdf_name"] = uploaded_file.name
        st.success(f"PDF '{uploaded_file.name}' processed and RAG pipeline ready!")
    else:
        st.error("Failed to set up RAG pipeline. Check OpenAI API key and logs.")

# Chat interface
if st.session_state["qa_chain"] and st.session_state["vectorstore"]:
    st.subheader("Ask a question about the PDF:")
    user_query = st.text_input("Your question:")

    if user_query:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state["qa_chain"].run(user_query)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}. Please try again or check logs.")
                if "Rate limit" in str(e):
                    st.warning("You might have hit the OpenAI API rate limit. Please wait a moment and try again.")
else:
    st.info("Please upload a PDF to start chatting.")

# Optional: Clear session state to re-upload new PDF
if st.button("Clear Chatbot & Upload New PDF"):
    st.session_state["qa_chain"] = None
    st.session_state["vectorstore"] = None
    st.session_state["current_pdf_name"] = None
    st.rerun()


#---- version prior to re-deployment----#
# import os
# from pathlib import Path
# import streamlit as st # Keep streamlit import if you use st.error/st.warning here

# from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, JSONLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents.agent_types import AgentType
# import pandas as pd

# # Define your document paths and directory here (assuming they are at the top of rag_core.py)
# # If not, ensure CHATBOT_DOCUMENTS and DOCUMENTS_DIRECTORY are defined or imported
# DOCUMENTS_DIRECTORY = "my_documents"
# CHATBOT_DOCUMENTS = {
#     "medical": ["llm_whisperer_LabTest_20250430.txt", "llm_whisperer_LabTest_20250520.txt", "llm_whisperer_LabTest_20250529.txt"],
#     "history": ["PhilHistory.txt"],
#     "finance": ["Sample-Financial-Statements-1.pdf"],
#     "inventory": ["inventory_data.csv"],
#     "legal": ["Registering Offense_ The Prohibition of Slurs as Trademarks.pdf"]
# }

# def get_inventory_dataframe(csv_path: Path):
#     """Loads inventory data from CSV into a Pandas DataFrame, ensuring numeric columns are correctly typed."""
#     try:
#         df = pd.read_csv(csv_path)
#         print(f"DEBUG: Successfully loaded inventory data into DataFrame. Rows: {len(df)}")

#         # Columns that should be numeric for calculations
#         # IMPORTANT: Ensure these names EXACTLY match your CSV headers (case-sensitive)
#         numeric_cols = ['Quantity_In_Stock', 'Unit_Cost_USD', 'Reorder_Level'] # Corrected: Reorder_Level (uppercase L)

#         for col in numeric_cols:
#             if col in df.columns:
#                 try:
#                     # Convert to numeric, coercing errors to NaN, then fill NaN with 0 or a sensible default
#                     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int if col == 'Quantity_In_Stock' else float)
#                     print(f"DEBUG: Converted column '{col}' to numeric (dtype: {df[col].dtype}).")
#                 except Exception as e:
#                     print(f"WARNING: Could not convert column '{col}' to numeric. Error: {e}")
#             else:
#                 print(f"WARNING: Numeric column '{col}' not found in {csv_path.name}. Skipping conversion.")
#         return df
#     except FileNotFoundError:
#         st.error(f"Error: CSV file not found at {csv_path}")
#         return None
#     except Exception as e:
#         st.error(f"Error loading or processing inventory CSV: {e}")
#         return None


# def setup_rag_pipeline(chatbot_name: str):
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # Using gpt-4o as default
#     embeddings = OpenAIEmbeddings()

#     # Determine document paths
#     doc_paths = CHATBOT_DOCUMENTS.get(chatbot_name)
#     if not doc_paths:
#         st.error(f"No document paths defined for chatbot '{chatbot_name}'. Check CHATBOT_DOCUMENTS.")
#         return None, None # Return tuple for consistency

#     # --- Document Loading and Splitting ---
#     # Load all documents for the selected chatbot
#     loaded_docs = []
#     print(f"Attempting to load documents for '{chatbot_name}':")
#     for doc_path in doc_paths:
#         full_path = Path(DOCUMENTS_DIRECTORY) / doc_path
#         print(f" - Loading: {full_path}...")
#         try:
#             # Loaders for different file types
#             if full_path.suffix == ".txt":
#                 loader = TextLoader(str(full_path))
#             elif full_path.suffix == ".pdf":
#                 loader = PyPDFLoader(str(full_path))
#             elif full_path.suffix == ".csv":
#                 loader = CSVLoader(str(full_path))
#             elif full_path.suffix == ".json":
#                 # Assuming simple JSON for now, might need custom parsing for complex JSON
#                 loader = JSONLoader(str(full_path), jq_schema='.', text_content=False)
#             else:
#                 print(f"WARNING: No loader configured for file type {full_path.suffix}")
#                 continue # Skip unsupported file types

#             docs = loader.load()
#             loaded_docs.extend(docs)
#             print(f"DEBUG: After loading {full_path}, {len(loaded_docs)} total documents extended into loaded_docs.")
#         except Exception as e:
#             st.error(f"Error loading document {full_path}: {e}")
#             print(f"ERROR: Error loading document {full_path}: {e}")
#             return None, None # Return tuple on error

#     if not loaded_docs:
#         st.error(f"No documents loaded for chatbot '{chatbot_name}'. Ensure files exist and are readable.")
#         return None, None # Return tuple for consistency
#     print(f"  Loaded {len(loaded_docs)} total documents so far.")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     print(f"DEBUG: split_documents received {len(loaded_docs)} documents to split.")
#     splits = text_splitter.split_documents(loaded_docs)
#     print(f"DEBUG: split_documents created {len(splits)} chunks.")

#     # --- Vector Store Setup (ChromaDB) ---
#     persist_directory = f'./chroma_db_{chatbot_name}'
#     if os.path.exists(persist_directory) and os.listdir(persist_directory):
#         print(f"DEBUG: Loading existing vector store from '{persist_directory}' for '{chatbot_name}'.")
#         vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#     else:
#         print(f"DEBUG: Creating new vector store in '{persist_directory}' for '{chatbot_name}'.")
#         vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
#         vector_store.persist() # Save the vector store to disk
#         print(f"DEBUG: Vector store created and persisted for '{chatbot_name}'.")

#     retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks

#     # --- RAG Chain Configuration ---
#     rag_chain = None
#     inventory_agent = None

#     if chatbot_name == "inventory":
#         # Load the CSV data into a Pandas DataFrame for the agent
#         inventory_df = get_inventory_dataframe(Path(DOCUMENTS_DIRECTORY) / "inventory_data.csv")
#         if inventory_df is None:
#             st.error("Failed to load inventory data for agent.")
#             return None, None

#         # Create the Pandas DataFrame Agent
#         inventory_agent = create_pandas_dataframe_agent(
#             llm,
#             inventory_df,
#             verbose=True,
#             agent_type=AgentType.OPENAI_FUNCTIONS,
#             handle_parsing_errors=True,
#             allow_dangerous_code=True
#         )

#         # For inventory, the RAG chain is still useful for general descriptive questions
#         inventory_rag_prompt_template = f"""
#         You are an Inventory Management AI Assistant.
#         Use the following pieces of retrieved context about inventory items to answer the question.
#         If the question cannot be answered from the provided context, state that you don't know but suggest checking the inventory data for specific numerical queries.
#         {{context}}
#         Question: {{input}}
#         """
#         prompt = ChatPromptTemplate.from_template(inventory_rag_prompt_template)
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, document_chain)

#     else: # For all other chatbots (history, medical, finance, legal)
#         # Define the prompt template using an f-string to insert the chatbot_name
#         # The {{context}} and {{input}} ensure these are treated as literal placeholders for LangChain.
#         prompt_template_str = f"""
#         You are a specialized AI assistant for {chatbot_name.capitalize()}.
#         Use the following pieces of retrieved context to answer the question.
#         If you don't know the answer, just say that you don't know, don't try to make up an answer.
#         {{context}}
#         Question: {{input}}
#         """
#         # This is the corrected line: Pass the f-string directly to from_template
#         prompt = ChatPromptTemplate.from_template(prompt_template_str)

#         document_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, document_chain)

#     print(f"DEBUG: RAG pipeline{' and Pandas Agent' if inventory_agent else ''} for '{chatbot_name}' setup complete.")
#     return rag_chain, inventory_agent # Always return the tuple