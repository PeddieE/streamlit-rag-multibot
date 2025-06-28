import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

# --- Constants ---
DOCUMENTS_DIRECTORY = "my_documents"
LLM_MODEL_NAME = "gpt-3.5-turbo" # Or "gpt-4" if you prefer
CHUNK_SIZE_TOKENS = 250
CHUNK_OVERLAP_TOKENS = 150
K_VALUE = 11

# Define file mappings for each chatbot type
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
    ]
}

# --- Functions (setup_rag_pipeline remains unchanged) ---
def setup_rag_pipeline(chatbot_name: str):
    faiss_index_path = f"faiss_index_{chatbot_name}"
    vector_store = None
    documents = []

    if os.path.exists(faiss_index_path) and os.listdir(faiss_index_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        files_to_load = CHATBOT_DOCUMENTS.get(chatbot_name)

        if not files_to_load:
            print(f"Error: No documents defined for '{chatbot_name}' chatbot. Please check CHATBOT_DOCUMENTS mapping.")
            return None

        for filename in files_to_load:
            file_path = Path(DOCUMENTS_DIRECTORY) / filename
            if not file_path.exists():
                print(f"Warning: File not found: '{file_path}'. Skipping this file.")
                continue

            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(str(file_path))
                elif filename.endswith(".csv"):
                    loader = CSVLoader(str(file_path))
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}. Skipping this file.")

        if not documents:
            print(f"Error: No documents were successfully loaded for '{chatbot_name}'. Cannot set up RAG pipeline.")
            return None

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            print(f"Error: No chunks were created for '{chatbot_name}'. Check chunking parameters or document content.")
            return None

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_documents(chunks, embeddings)

        os.makedirs(faiss_index_path, exist_ok=True)
        vector_store.save_local(faiss_index_path)

    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2)

    if chatbot_name == "medical":
        prompt_template = """
        You are a highly knowledgeable medical AI assistant. Your goal is to answer questions based *only* on the provided lab test results and medical reports.
        If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
        Do not make up information or speculate.
        If the user asks for results over different dates, compare them and provide all available results.

        Context:
        {context}

        Question: {input}
        """
    elif chatbot_name == "inventory":
        prompt_template = """
        You are an inventory management AI assistant. Your goal is to answer questions about product stock, costs, and suppliers based *only* on the provided inventory data.
        If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
        Do not make up information or speculate.
        When asked about stock levels, reorder levels, or costs, provide the exact numbers.

        Context:
        {context}

        Question: {input}
        """
    elif chatbot_name == "history":
        prompt_template = """
        You are a Philippine history expert AI assistant. Your goal is to answer questions about Philippine history based *only* on the provided historical documents.
        If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
        Do not make up information or speculate.

        Context:
        {context}

        Question: {input}
        """
    elif chatbot_name == "finance":
        prompt_template = """
        You are a financial analysis AI assistant. Your goal is to answer questions about financial statements and reports based *only* on the provided financial documents.
        If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
        Do not make up information or speculate.
        When asked for specific financial figures (e.g., revenue, net income), provide the exact numbers from the document.

        Context:
        {context}

        Question: {input}
        """
    else:
        prompt_template = """
        You are a helpful AI assistant. Your goal is to answer questions based *only* on the provided context.
        If the answer is not present in the provided context, state that you don't have enough information to provide an answer.
        Do not make up information or speculate.

        Context:
        {context}

        Question: {input}
        """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": K_VALUE}),
        document_chain
    )
    return retrieval_chain

def main():
    rag_chains = {}
    available_chatbots = list(CHATBOT_DOCUMENTS.keys())

    current_chatbot_name = None
    current_rag_chain = None

    while True:
        if current_chatbot_name is None:
            print("\n--- RAG App Interactive Mode ---")
            print("Please select a chatbot to start or continue interaction:")
            for i, bot_name in enumerate(available_chatbots):
                print(f"{i+1}. {bot_name.capitalize()} Chatbot")
            # --- UPDATED PROMPT FOR EXIT ---
            print("Type 'E' to exit.")
            # --- END UPDATED PROMPT ---

            choice = input("Your choice: ").lower()
            # --- UPDATED EXIT CONDITION ---
            if choice == 'exit' or choice == 'quit' or choice == 'e':
                print("Exiting application. Goodbye!")
                break
            # --- END UPDATED EXIT CONDITION ---
            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(available_chatbots):
                    current_chatbot_name = available_chatbots[choice_index]
                    if current_chatbot_name not in rag_chains:
                        rag_chains[current_chatbot_name] = setup_rag_pipeline(current_chatbot_name)
                    else:
                        pass

                    current_rag_chain = rag_chains[current_chatbot_name]

                    if current_rag_chain is None:
                        print(f"ERROR: Failed to set up or retrieve {current_chatbot_name} chatbot pipeline. Please check file paths and content.")
                        print("Resetting selection. Please choose another chatbot or try again.")
                        current_chatbot_name = None
                else:
                    print("Invalid choice number. Please enter a number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number (e.g., 1, 2) or 'E'.") # Updated error message
            continue

        # --- UPDATED PROMPT FOR QUESTION INPUT ---
        user_question = input(f"\n{current_chatbot_name.capitalize()} question (Type 'S' for switch, 'E' for exit): ").lower() # Made lower here directly
        # --- END UPDATED PROMPT ---

        # --- UPDATED EXIT AND SWITCH CONDITIONS ---
        if user_question in ['exit', 'quit', 'e']:
            print("Exiting application. Goodbye!")
            break
        elif user_question in ['switch', 's']:
            print("Switching chatbots...")
            current_chatbot_name = None
            current_rag_chain = None
            continue
        # --- END UPDATED CONDITIONS ---

        try:
            response = current_rag_chain.invoke({"input": user_question})
            print(f"\nAI Assistant ({current_chatbot_name.capitalize()}):")
            print(response["answer"])

        except Exception as e:
            print(f"An error occurred during retrieval or generation: {e}")
            print("Please try your question again.")

if __name__ == "__main__":
    main()

        
     
    