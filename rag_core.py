#version for re-deployment

import os
from pathlib import Path
import streamlit as st # Keep streamlit import if you use st.error/st.warning here

from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd

# Define your document paths and directory here (assuming they are at the top of rag_core.py)
# If not, ensure CHATBOT_DOCUMENTS and DOCUMENTS_DIRECTORY are defined or imported
DOCUMENTS_DIRECTORY = "my_documents"
CHATBOT_DOCUMENTS = {
    "medical": ["llm_whisperer_LabTest_20250430.txt", "llm_whisperer_LabTest_20250520.txt", "llm_whisperer_LabTest_20250529.txt"],
    "history": ["PhilHistory.txt"],
    "finance": ["Sample-Financial-Statements-1.pdf"],
    "inventory": ["inventory_data.csv"],
    "legal": ["Registering Offense_ The Prohibition of Slurs as Trademarks.pdf"]
}

# --- START OF NEW MODIFICATIONS FOR PERSISTENT DISK ---

# Get the base path for ChromaDB persistence from an environment variable.
# On Render, we will set CHROMA_PERSIST_BASE_DIR to your disk's mount path (e.g., "/data").
# Locally, it will default to '.' (the current directory) for development.
CHROMA_PERSIST_BASE_DIR = os.getenv("CHROMA_PERSIST_BASE_DIR", ".")

# Define the directory where ALL ChromaDB collections will be stored within the persistent base path
# This will create something like:
#   - './chroma_data_storage' locally
#   - '/data/chroma_data_storage' on Render (assuming /data is your mount path)
CHROMA_DB_STORAGE_FOLDER = os.path.join(CHROMA_PERSIST_BASE_DIR, "chroma_data_storage")

# Ensure the directory exists (important for both local and Render environments)
# This will create the folder on the persistent disk if it doesn't exist.
os.makedirs(CHROMA_DB_STORAGE_FOLDER, exist_ok=True)

# Add a debug print to see the actual path being used in Render logs
print(f"DEBUG: ChromaDB will attempt to persist data in: {CHROMA_DB_STORAGE_FOLDER}")

# --- END OF NEW MODIFICATIONS ---


def get_inventory_dataframe(csv_path: Path):
    """Loads inventory data from CSV into a Pandas DataFrame, ensuring numeric columns are correctly typed."""
    try:
        df = pd.read_csv(csv_path)
        print(f"DEBUG: Successfully loaded inventory data into DataFrame. Rows: {len(df)}")

        # Columns that should be numeric for calculations
        # IMPORTANT: Ensure these names EXACTLY match your CSV headers (case-sensitive)
        numeric_cols = ['Quantity_In_Stock', 'Unit_Cost_USD', 'Reorder_Level'] # Corrected: Reorder_Level (uppercase L)

        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Convert to numeric, coercing errors to NaN, then fill NaN with 0 or a sensible default
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int if col == 'Quantity_In_Stock' else float)
                    print(f"DEBUG: Converted column '{col}' to numeric (dtype: {df[col].dtype}).")
                except Exception as e:
                    print(f"WARNING: Could not convert column '{col}' to numeric. Error: {e}")
            else:
                print(f"WARNING: Numeric column '{col}' not found in {csv_path.name}. Skipping conversion.")
        return df
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading or processing inventory CSV: {e}")
        return None


def setup_rag_pipeline(chatbot_name: str):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # Using gpt-4o as default
    embeddings = OpenAIEmbeddings()

    # Determine document paths
    doc_paths = CHATBOT_DOCUMENTS.get(chatbot_name)
    if not doc_paths:
        st.error(f"No document paths defined for chatbot '{chatbot_name}'. Check CHATBOT_DOCUMENTS.")
        return None, None # Return tuple for consistency

    # --- Document Loading and Splitting ---
    # Load all documents for the selected chatbot
    loaded_docs = []
    print(f"Attempting to load documents for '{chatbot_name}':")
    for doc_path in doc_paths:
        full_path = Path(DOCUMENTS_DIRECTORY) / doc_path
        print(f" - Loading: {full_path}...")
        try:
            # Loaders for different file types
            if full_path.suffix == ".txt":
                loader = TextLoader(str(full_path))
            elif full_path.suffix == ".pdf":
                loader = PyPDFLoader(str(full_path))
            elif full_path.suffix == ".csv":
                loader = CSVLoader(str(full_path))
            elif full_path.suffix == ".json":
                # Assuming simple JSON for now, might need custom parsing for complex JSON
                loader = JSONLoader(str(full_path), jq_schema='.', text_content=False)
            else:
                print(f"WARNING: No loader configured for file type {full_path.suffix}")
                continue # Skip unsupported file types

            docs = loader.load()
            loaded_docs.extend(docs)
            print(f"DEBUG: After loading {full_path}, {len(loaded_docs)} total documents extended into loaded_docs.")
        except Exception as e:
            st.error(f"Error loading document {full_path}: {e}")
            print(f"ERROR: Error loading document {full_path}: {e}")
            return None, None # Return tuple on error

    if not loaded_docs:
        st.error(f"No documents loaded for chatbot '{chatbot_name}'. Ensure files exist and are readable.")
        return None, None # Return tuple for consistency
    print(f"   Loaded {len(loaded_docs)} total documents so far.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print(f"DEBUG: split_documents received {len(loaded_docs)} documents to split.")
    splits = text_splitter.split_documents(loaded_docs)
    print(f"DEBUG: split_documents created {len(splits)} chunks.")

    # --- Vector Store Setup (ChromaDB) ---
    # --- START OF MODIFICATION ---
    # Define the specific persistence directory for this chatbot's collection
    # It will be inside the main CHROMA_DB_STORAGE_FOLDER
    # Example: /data/chroma_data_storage/chroma_db_medical
    persist_directory = os.path.join(CHROMA_DB_STORAGE_FOLDER, f'chroma_db_{chatbot_name}')
    print(f"DEBUG: Using ChromaDB persist_directory: {persist_directory}") # Another useful debug print

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"DEBUG: Loading existing vector store from '{persist_directory}' for '{chatbot_name}'.")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print(f"DEBUG: Creating new vector store in '{persist_directory}' for '{chatbot_name}'.")
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
        vector_store.persist() # Save the vector store to disk
        print(f"DEBUG: Vector store created and persisted for '{chatbot_name}'.")
    # --- END OF MODIFICATION ---

    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks

    # --- RAG Chain Configuration ---
    rag_chain = None
    inventory_agent = None

    if chatbot_name == "inventory":
        # Load the CSV data into a Pandas DataFrame for the agent
        inventory_df = get_inventory_dataframe(Path(DOCUMENTS_DIRECTORY) / "inventory_data.csv")
        if inventory_df is None:
            st.error("Failed to load inventory data for agent.")
            return None, None

        # Create the Pandas DataFrame Agent
        inventory_agent = create_pandas_dataframe_agent(
            llm,
            inventory_df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )

        # For inventory, the RAG chain is still useful for general descriptive questions
        inventory_rag_prompt_template = f"""
        You are an Inventory Management AI Assistant.
        Use the following pieces of retrieved context about inventory items to answer the question.
        If the question cannot be answered from the provided context, state that you don't know but suggest checking the inventory data for specific numerical queries.
        {{context}}
        Question: {{input}}
        """
        prompt = ChatPromptTemplate.from_template(inventory_rag_prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

    else: # For all other chatbots (history, medical, finance, legal)
        # Define the prompt template using an f-string to insert the chatbot_name
        # The {{context}} and {{input}} ensure these are treated as literal placeholders for LangChain.
        prompt_template_str = f"""
        You are a specialized AI assistant for {chatbot_name.capitalize()}.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {{context}}
        Question: {{input}}
        """
        # This is the corrected line: Pass the f-string directly to from_template
        prompt = ChatPromptTemplate.from_template(prompt_template_str)

        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

    print(f"DEBUG: RAG pipeline{' and Pandas Agent' if inventory_agent else ''} for '{chatbot_name}' setup complete.")
    return rag_chain, inventory_agent # Always return the tuple


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