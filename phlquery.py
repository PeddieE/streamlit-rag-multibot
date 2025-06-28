# phlquery.py - non-interactive version of the script; questions are
#               embedded within the code.
# Load the API key
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAI # The corrected import


# Load the document
#loaded_document = TextLoader("./data/PhilHistory.txt", encoding='utf-8').load()
loaded_document = TextLoader("./data/llm_whisperer_LabTest_20250529.txt", encoding='utf-8').load()
# If you want to see the document content, uncomment the next line:
# print(loaded_document)

# Split text into chunks and create vector database
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks_of_text = text_splitter.split_documents(loaded_document)

embeddings = OpenAIEmbeddings() # This line was duplicated, kept only one
vector_db = FAISS.from_documents(chunks_of_text, embeddings) # This line was duplicated, kept only one

# Optional: print lengths or objects if needed for debugging, but not for final script
# print(len(chunks_of_text))
# print(vector_db)

retriever = vector_db.as_retriever(search_kwargs={"k":3})

# Optional: print retriever object if needed for debugging
# print(retriever)

# Example of initial retrieval response (without LLM)
# response = retriever.invoke("How many islands does the Philippines have?")
# print(response) # You'd need a print statement to see it in a .py file

# Get the final answer using a simple LCEL
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """ Answer the question based only on the ff: context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI() # Uses ChatOpenAI for the final chain

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# # Example invocations with the chain - for CHATBOt philippine history
# response_1 = chain.invoke("How many islands does the Philippines have?")
# print("Response 1:", response_1)

# response_2 = chain.invoke("When was Jose Rizal executed?")
# print("Response 2:", response_2)

# response_3 = chain.invoke("Explain the galleon trade")
#print("Response 3:", response_3)


# Example invocations with the chain - for lab test result
response_1 = chain.invoke("Show me my eGFR")
print("Response 1:", response_1)

response_2 = chain.invoke("Show me my creatinine")
print("Response 2:", response_2)

response_3 = chain.invoke("show my my fasting blood sugar")
print("Response 3:", response_3)

response_4 = chain.invoke("show my my lipid profile")
print("Response 4:", response_4)