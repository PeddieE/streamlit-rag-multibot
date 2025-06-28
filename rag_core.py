# Script name: rag_code.py
# Description: chatbots for Finance, Medical, Inventory and History all combined
# Author: Siegfried Manuel R. Eata

import os
from pathlib import Path

# --- LangChain Document Loaders ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader # Added JSONLoader
)

# --- LangChain Components ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Vector Store: ChromaDB ---
from langchain_community.vectorstores import Chroma


# --- Configuration Constants ---
DOCUMENTS_DIRECTORY = "./my_documents" # Path to your folder containing source documents
PERSIST_DIRECTORY = "./chroma_db"     # Directory where ChromaDB data will be stored locally

# Define specific documents for each chatbot's knowledge base
# Ensure the paths are correct relative to DOCUMENTS_DIRECTORY
# Updated with common file types: PDF, TXT, CSV, JSON
CHATBOT_DOCUMENTS = {
    "medical": [
        "llm_whisperer_LabTest_20250430.txt",
        "llm_whisperer_LabTest_20250520.txt",
        "llm_whisperer_LabTest_20250529.txt"
    ],
    "inventory": [
        "inventory_data.csv"
    ],
    "history": [
        "PhilHistory.txt"
 
    ],
    "finance": [
        "Sample-Financial-Statements-1.pdf"
       
    ],

    "legal": [
        "fictional_court_case.txt",
        "fictional_contract_clause.txt",
        "fictional_regulation_overview.txt"
       
    ]
     
    # Add more chatbots and their respective document paths here

    
}

# --- Document Loading and Processing Utilities ---

def get_loader(file_path: Path):
    """
    Returns the appropriate Langchain document loader based on file extension.
    Supports .pdf, .txt, .json, .csv.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    file_extension = file_path.suffix.lower()
    if file_extension == ".pdf":
        return PyPDFLoader(str(file_path))
    elif file_extension == ".txt":
        return TextLoader(str(file_path))
    elif file_extension == ".csv":
        return CSVLoader(str(file_path))
    elif file_extension == ".json":
        # For JSONLoader, you often need to define a jq_schema.
        # '.' means load the entire JSON structure.
        # You might need to adjust this schema based on your specific JSON file structure.
        # For example, if your JSON is an array of objects and you want specific fields:
        # jq_schema='.[].text_field', or if it's a single object: jq_schema='.data.article'
        # For general text extraction, '.' often works.
        return JSONLoader(file_path=str(file_path), jq_schema='.', text_content=False)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Please use .pdf, .txt, .csv, or .json.")

def load_documents_for_chatbot(chatbot_name: str):
    """
    Loads documents specified for a given chatbot from the CHATBOT_DOCUMENTS mapping.
    Handles errors for missing files or unsupported types.
    """
    documents_to_load = CHATBOT_DOCUMENTS.get(chatbot_name)
    if not documents_to_load:
        print(f"Warning: No documents defined for chatbot: '{chatbot_name}'.")
        return []

    loaded_docs = []
    print(f"Attempting to load documents for '{chatbot_name}':")
    for doc_name in documents_to_load:
        file_path = Path(DOCUMENTS_DIRECTORY) / doc_name
        try:
            print(f"  - Loading: {file_path}...")
            loader = get_loader(file_path)
            loaded_docs.extend(loader.load())
            print(f"    Loaded {len(loaded_docs)} total documents so far.")
        except FileNotFoundError as e:
            print(f"  - Error: {e}")
        except ValueError as e: # For unsupported file types from get_loader
            print(f"  - Error loading {file_path}: {e}")
        except Exception as e: # Catch any other unexpected errors during loading
            print(f"  - An unexpected error occurred while loading {file_path}: {e}")
    
    if not loaded_docs:
        print(f"No documents were successfully loaded for '{chatbot_name}'.")
    return loaded_docs

def split_documents(documents):
    """
    Splits loaded LangChain Document objects into smaller, manageable chunks.
    Uses RecursiveCharacterTextSplitter which is robust for various text types.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Maximum number of tokens per chunk
        chunk_overlap=200    # Overlap between consecutive chunks to maintain context
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        print("Warning: No text chunks were created from the documents. Check chunking parameters or document content.")
    return chunks

# --- RAG Pipeline Setup ---

def setup_rag_pipeline(chatbot_name: str):
    """
    Configures and returns a LangChain RAG pipeline for a specific chatbot.
    It handles loading/creating the ChromaDB vector store, setting up the LLM,
    and defining the retrieval chain with a tailored prompt.
    """
    try:
        # 1. Initialize Embedding Model and Language Model
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002") # Recommended OpenAI embedding model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2) # Use gpt-3.5-turbo for cost-efficiency; adjust temperature

        # Define the specific persist directory for this chatbot's ChromaDB instance
        # This keeps each chatbot's vector store separate
        chatbot_persist_dir = Path(PERSIST_DIRECTORY) / chatbot_name
        
        # 2. Load or Create ChromaDB Vector Store
        vector_store = None
        # Check if the ChromaDB directory exists and contains data for this chatbot
        if chatbot_persist_dir.exists() and any(chatbot_persist_dir.iterdir()):
            print(f"🔄 Loading existing ChromaDB for '{chatbot_name}' from '{chatbot_persist_dir}'...")
            # When loading an existing ChromaDB, you MUST pass the same embedding function
            # that was used to create it.
            vector_store = Chroma(
                persist_directory=str(chatbot_persist_dir),
                embedding_function=embedding_model # Pass the embedding model
            )
            print(f"✅ ChromaDB for '{chatbot_name}' loaded successfully with {vector_store._collection.count()} entries.")
        else:
            print(f"✨ Creating new ChromaDB for '{chatbot_name}'...")
            # Ensure the parent persist directory exists
            Path(PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)

            documents = load_documents_for_chatbot(chatbot_name)
            if not documents:
                print(f"🛑 Error: No documents found or loaded for '{chatbot_name}'. Cannot build RAG pipeline without data.")
                return None

            chunks = split_documents(documents)
            if not chunks:
                print(f"🛑 Error: No text chunks were created from documents for '{chatbot_name}'. Cannot build RAG pipeline.")
                return None
            
            # Create the ChromaDB from chunks and persist it to the specified directory
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=str(chatbot_persist_dir) # Save to this specific directory
            )
            # Explicitly persist the data to disk after creation
            # Note: .persist() is typically called when you add more data or at the end of a session
            # if you're not using .from_documents directly with persist_directory
            # For .from_documents with persist_directory, it's often handled internally,
            # but explicit call doesn't hurt.
            vector_store.persist() 
            print(f"✅ ChromaDB for '{chatbot_name}' created and persisted with {vector_store._collection.count()} entries.")


        # 3. Define Prompt Template based on Chatbot Name
        # These prompts guide the LLM's behavior and response style for each domain.
        if chatbot_name == "medical":
            prompt_template = """
            You are a highly knowledgeable Medical AI Assistant. Your primary goal is to answer questions based *only* on the provided medical lab test results and patient records.
            If the answer cannot be found directly in the context, state that you don't have enough information to provide an answer.
            Do not guess, make up information, or speculate.
            When comparing results over different dates, provide all available data for context.

            Context:
            {context}

            Question: {input}
            """
        elif chatbot_name == "inventory":
            prompt_template = """
            You are an Inventory Management AI Assistant. Your goal is to answer questions about product stock levels, costs, suppliers, and reorder points based *only* on the provided inventory data and supplier contacts.
            If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
            Do not make up information or speculate.
            Always provide exact numbers when asked about stock, costs, or quantities.

            Context:
            {context}

            Question: {input}
            """
        elif chatbot_name == "history":
            prompt_template = """
            You are a World History Expert AI Assistant. Your goal is to answer questions about historical events, figures, and periods based *only* on the provided historical documents.
            If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
            Do not make up information or speculate.
            Focus on factual accuracy derived strictly from the given texts.

            Context:
            {context}

            Question: {input}
            """
        elif chatbot_name == "finance":
            prompt_template = """
            You are a Financial Analysis AI Assistant. Your goal is to answer questions about financial statements, market trends, and economic data based *only* on the provided financial reports and analysis documents.
            If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
            Do not make up information or speculate.
            Provide specific financial figures and data points directly from the context when requested.

            Context:
            {context}

            Question: {input}
            """
        else: # Default prompt for any unspecified chatbot name
            prompt_template = """
            You are a helpful AI Assistant. Your goal is to answer questions based *only* on the provided context.
            If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
            Do not make up information or speculate.

            Context:
            {context}

            Question: {input}
            """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # 4. Create Document Chain (stuffs context into the prompt)
        document_chain = create_stuff_documents_chain(llm, prompt)

        # 5. Create Retrieval Chain (combines retriever and document chain)
        # We can specify the number of relevant documents to retrieve (k)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant documents
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        print(f"🤖 RAG pipeline for '{chatbot_name}' initialized successfully.")
        return retrieval_chain

    except Exception as e:
        print(f"❌ An error occurred during RAG pipeline setup for '{chatbot_name}': {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed traceback during debugging
        return None

# --- Remember to remove the if __name__ == "__main__": main() block
# from this file if you are using it with Streamlit.
# Streamlit will import and call these functions directly.