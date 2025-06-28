import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# Define the directory where your pre-processed text files are stored
DATA_DIR = "my_documents"

# Configure chunking parameters
# Using token-based splitter for better LLM compatibility
# You can adjust chunk_size and chunk_overlap based on your testing.
# For 2500 characters, it's roughly 625 tokens (2500 / 4).
# I'll set it to 650 tokens as a good starting point for token-based splitting.
CHUNK_SIZE_TOKENS = 650
CHUNK_OVERLAP_TOKENS = 100

# --- Helper Function for RAG Setup ---
def setup_rag_pipeline():
    """
    Sets up the RAG pipeline by loading multiple documents, chunking them,
    creating embeddings, and building a FAISS vector store.
    """
    print(f"Loading documents from directory: {DATA_DIR}...")
    documents = []
    # Loop through all text files in the specified directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                print(f"  Loaded: {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue # Skip to the next file if there's an error

    if not documents:
        print(f"No text documents found in '{DATA_DIR}'. Please ensure you have .txt files there.")
        print("Exiting application as no documents are available for RAG.")
        exit()

    print(f"Total documents loaded: {len(documents)}")

    # Initialize text splitter (token-based)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", # For gpt-4, gpt-3.5-turbo models
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", " ", ""]
    )

    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Example chunk (first 200 chars): {chunks[0].page_content[:200]}...")

    # Initialize embeddings model
    print("Creating embeddings and building FAISS vector store. This might take a moment...")
    embeddings = OpenAIEmbeddings()

    # Create the FAISS vector store from all chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("FAISS vector store built successfully!")

    # Initialize LLM
    print("Initializing LLM (ChatOpenAI)...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # You can choose other models like "gpt-4"

    # Create the prompt template for the RAG chain
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant specialized in medical lab results and reports.
    Answer the user's question based ONLY on the provided context.
    If the answer cannot be found in the context, politely state that you don't have enough information.

    Context:
    {context}

    Question: {input}
    """)

    # Create the document combining chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    #retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)
    retrieval_chain = create_retrieval_chain(
            vector_store.as_retriever(search_kwargs={"k": 30}), # This sets k to 10
                document_chain
    )

    print("RAG pipeline setup complete. Ready to answer questions!")
    return retrieval_chain

# --- Main Application Logic ---
if __name__ == "__main__":
    rag_chain = setup_rag_pipeline()

    print("\n--- RAG App Interactive Mode ---")
    print("Type your questions below. Type 'exit' or 'quit' to end the session.")

    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting RAG app. Goodbye!")
            break

        print("Searching and generating answer...")
        try:
            response = rag_chain.invoke({"input": user_question})
            # The structure of the response from create_retrieval_chain is a dict
            # with 'input', 'context', and 'answer' keys.
            print("\nAI Assistant:")
            print(response["answer"])

            # Optionally, print the retrieved context for debugging/transparency
            print("\n--- Retrieved Context ---")
            for i, doc in enumerate(response["context"]):
                print(f"Chunk {i+1} (Source: {doc.metadata.get('source', 'N/A')}):\n{doc.page_content[:300]}...") # Print first 300 chars
            print("-------------------------")

        except Exception as e:
            print(f"An error occurred during retrieval or generation: {e}")
            print("Please try your question again.")


















#----------------------- this is the original code ----------------------------------
#phlquery_interactive.py <- interactive version of the script
#                           where users provide the questions during runtime

# import os
# from dotenv import load_dotenv, find_dotenv

# _ = load_dotenv(find_dotenv())
# openai_api_key = os.environ["OPENAI_API_KEY"]

# # Import necessary LangChain components
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # --- Document Loading and Vector Store Setup ---
# print("Loading document and setting up vector database...")
# loaded_document = TextLoader("./data/PhilHistory.txt", encoding='utf-8').load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# chunks_of_text = text_splitter.split_documents(loaded_document)

# embeddings = OpenAIEmbeddings()
# vector_db = FAISS.from_documents(chunks_of_text, embeddings)
# retriever = vector_db.as_retriever(search_kwargs={"k":3})
# print("Vector database setup complete.\n")

# # --- LangChain Expression Language (LCEL) Chain Setup ---
# template = """ Answer the question based only on the ff: context:

# {context}

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)
# model = ChatOpenAI(temperature=0.0) # Added temperature for more consistent answers if preferred

# def format_docs(docs):
#     return "\n\n".join([d.page_content for d in docs])

# chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
# )

# # --- Interactive Loop for Questions ---
# print("RAG System Ready! Type your questions below.")
# print("Type 'exit' or 'quit' to end the session.")

# while True:
#     user_question = input("\nYour Question: ") # Get input from the user

#     if user_question.lower() in ["exit", "quit"]:
#         print("Exiting RAG session. Goodbye!")
#         break # Exit the loop

#     try:
#         response = chain.invoke(user_question)
#         print("\nAI Response:", response)
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         print("Please try again or check your API key/internet connection.")
