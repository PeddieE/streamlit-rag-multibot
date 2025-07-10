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
from langchain.agents import AgentExecutor, Tool
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import MessagesPlaceholder
import pandas as pd

# Define your document paths and directory here (assuming they are at the top of rag_core.py)
DOCUMENTS_DIRECTORY = "my_documents"
CHATBOT_DOCUMENTS = {
    "medical": ["llm_whisperer_LabTest_20250430.txt", "llm_whisperer_LabTest_20250520.txt", "llm_whisperer_LabTest_20250529.txt"],
    "history": ["PhilHistory.txt"],
    "finance": ["Sample-Financial-Statements-1.pdf"],
    "inventory": ["inventory_data.csv"],
    "legal": ["Registering Offense_ The Prohibition of Slurs as Trademarks.pdf"]
}

# --- START OF NEW MODIFICATIONS FOR PERSISTENT DISK ---
CHROMA_PERSIST_BASE_DIR = os.getenv("CHROMA_PERSIST_BASE_DIR", ".")
CHROMA_DB_STORAGE_FOLDER = os.path.join(CHROMA_PERSIST_BASE_DIR, "chroma_data_storage")
os.makedirs(CHROMA_DB_STORAGE_FOLDER, exist_ok=True)
print(f"DEBUG: ChromaDB will attempt to persist data in: {CHROMA_DB_STORAGE_FOLDER}")
# --- END OF NEW MODIFICATIONS ---


def get_inventory_dataframe(csv_path: Path):
    """Loads inventory data from CSV into a Pandas DataFrame, ensuring numeric columns are correctly typed."""
    try:
        df = pd.read_csv(csv_path)
        print(f"DEBUG: Successfully loaded inventory data into DataFrame. Rows: {len(df)}")

        numeric_cols = ['Quantity_In_Stock', 'Unit_Cost_USD', 'Reorder_Level']

        for col in numeric_cols:
            if col in df.columns:
                try:
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
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings()

    doc_paths = CHATBOT_DOCUMENTS.get(chatbot_name)
    if not doc_paths:
        st.error(f"No document paths defined for chatbot '{chatbot_name}'. Check CHATBOT_DOCUMENTS.")
        return None

    loaded_docs = []
    print(f"Attempting to load documents for '{chatbot_name}':")
    for doc_path in doc_paths:
        full_path = Path(DOCUMENTS_DIRECTORY) / doc_path
        print(f" - Loading: {full_path}...")
        try:
            if full_path.suffix == ".txt":
                loader = TextLoader(str(full_path))
            elif full_path.suffix == ".pdf":
                loader = PyPDFLoader(str(full_path))
            elif full_path.suffix == ".csv":
                loader = CSVLoader(str(full_path))
            elif full_path.suffix == ".json":
                loader = JSONLoader(str(full_path), jq_schema='.', text_content=False)
            else:
                print(f"WARNING: No loader configured for file type {full_path.suffix}")
                continue

            docs = loader.load()
            loaded_docs.extend(docs)
            print(f"DEBUG: After loading {full_path}, {len(loaded_docs)} total documents extended into loaded_docs.")
        except Exception as e:
            st.error(f"Error loading document {full_path}: {e}")
            print(f"ERROR: Error loading document {full_path}: {e}")
            return None

    if not loaded_docs:
        st.error(f"No documents loaded for chatbot '{chatbot_name}'. Ensure files exist and are readable.")
        return None
    print(f"    Loaded {len(loaded_docs)} total documents so far.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print(f"DEBUG: split_documents received {len(loaded_docs)} documents to split.")
    splits = text_splitter.split_documents(loaded_docs)
    print(f"DEBUG: split_documents created {len(splits)} chunks.")

    # --- Vector Store Setup (ChromaDB) ---
    persist_directory = os.path.join(CHROMA_DB_STORAGE_FOLDER, f'chroma_db_{chatbot_name}')
    print(f"DEBUG: Using ChromaDB persist_directory: {persist_directory}")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"DEBUG: Loading existing vector store from '{persist_directory}' for '{chatbot_name}'.")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print(f"DEBUG: Creating new vector store in '{persist_directory}' for '{chatbot_name}'.")
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
        vector_store.persist()
        print(f"DEBUG: Vector store created and persisted for '{chatbot_name}'.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # --- RAG Chain Configuration ---
    # Always create the RAG chain, as it can be a tool or the primary chain
    if chatbot_name == "inventory":
        rag_prompt_template = f"""
        You are an Inventory Management AI Assistant.
        Use the following pieces of retrieved context about inventory items to answer the question.
        If the question cannot be answered from the provided context, state that you don't know.
        For questions requiring data analysis or numerical computations on inventory items (e.g., 'total cost', 'average quantity', 'find products with quantity less than X'),
        **DO NOT try to calculate yourself**. Refer to the 'inventory_data_analysis' tool.
        {{context}}
        Question: {{input}}
        """
    else: # For all other chatbots (history, medical, finance, legal)
        # Define the prompt template using an f-string to insert the chatbot_name
        # The {{context}} and {{input}} ensure these are treated as literal placeholders for LangChain.
        rag_prompt_template = f"""
        You are a specialized AI assistant for {chatbot_name.capitalize()}.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {{context}}
        Question: {{input}}
        """
    prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)


    # --- Agent Setup for 'inventory' Chatbot ---
    if chatbot_name == "inventory":
        inventory_df = get_inventory_dataframe(Path(DOCUMENTS_DIRECTORY) / "inventory_data.csv")
        if inventory_df is None:
            st.error("Failed to load inventory data for agent.")
            return None

        pandas_agent_tool_executor = create_pandas_dataframe_agent( # Renamed for clarity
            llm,
            inventory_df,
            verbose=True, # This verbose is for the pandas agent's internal steps
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True, # UserWarning will still show, but functionality should be fine
            allow_dangerous_code=True
        )

        tools = [
            Tool(
                name="inventory_data_analysis",
                func=pandas_agent_tool_executor.invoke, # The Pandas Agent as a tool
                description="""Useful for answering questions about inventory data that require computations, aggregations, filtering, or numerical analysis.
                Input should be a question about the inventory data, e.g., 'What is the total quantity of 'Electronics' products?',
                'List products with quantity less than 50', 'Compute the average unit cost.', 'What is the sum of Quantity_In_Stock for all products?'"""
            ),
            Tool(
                name="inventory_information_retriever",
                func=rag_chain.invoke, # The RAG chain as a tool for general info retrieval
                description="""Useful for answering general informational questions about inventory items that can be found directly in the documents.
                Do not use for questions requiring calculations or data analysis. Input should be a question to retrieve information."""
            )
        ]

        # Define the main agent prompt for the AgentExecutor
        # This prompt is crucial for guiding the agent's decision-making over the tools
        agent_executor_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant that manages inventory data. You have access to two tools: 'inventory_data_analysis' for numerical queries and 'inventory_information_retriever' for general information from documents. Use the most appropriate tool for the user's question."),
                MessagesPlaceholder(variable_name="chat_history", optional=True), # Optional: for future chat history
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the actual agent object
        agent = create_openai_functions_agent(llm, tools, agent_executor_prompt)

        # Initialize the AgentExecutor with the created agent
        main_agent_executor = AgentExecutor(
            agent=agent, # Pass the actual agent object here
            tools=tools,
            verbose=True, # <--- THIS IS THE MAIN AGENT'S VERBOSE, CRITICAL FOR SEEING TOOL SELECTION
            handle_parsing_errors=True,
            # chat_history will be handled by the prompt or the calling application if needed
        )
        print(f"DEBUG: Inventory AgentExecutor setup complete.")
        return main_agent_executor # Return the orchestrating agent

    else: # For all other chatbots (history, medical, finance, legal)
        print(f"DEBUG: RAG pipeline for '{chatbot_name}' setup complete.")
        return rag_chain # Return only the RAG chain for non-inventory bots