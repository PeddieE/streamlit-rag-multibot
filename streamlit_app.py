import streamlit as st
import os
from pathlib import Path

# Import the core functions and constants from your rag_core.py file
from rag_core import setup_rag_pipeline, CHATBOT_DOCUMENTS, DOCUMENTS_DIRECTORY

# --- CSS LOADING ---
# Function to load CSS from an external file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file from the .streamlit directory
local_css(os.path.join(".streamlit", "style.css"))
# --- END CSS LOADING ---


# Set the page configuration
st.set_page_config(
    page_title="Multi-Chatbot RAG App",
    page_icon="🤖",
    layout="centered" # or "wide" for more space
)

# --- App Title and Description ---
st.title("👨‍💻 Multi-Chatbot RAG Application")
st.markdown("Select a specialized chatbot from the sidebar and ask your questions!")
st.info("The knowledge bases for each chatbot are built from your local documents.")

# --- Session State Initialization ---
# This ensures that the RAG chains/Agents and chat history persist across reruns
if 'rag_pipelines' not in st.session_state: # Renamed from 'rag_chains' for clarity
    st.session_state['rag_pipelines'] = {}
if 'current_chatbot_name' not in st.session_state:
    st.session_state['current_chatbot_name'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


# --- Sidebar for Chatbot Selection ---
with st.sidebar:
    st.header("Chatbot Selection")

    available_chatbots = sorted(list(CHATBOT_DOCUMENTS.keys())) # Sort for consistent display

    display_options = ["Select a Chatbot"] + \
                      [
                          "Philippine History" if bot == "history" else bot.capitalize()
                          for bot in available_chatbots
                      ]

    selected_chatbot_display = st.selectbox(
        "Choose a Chatbot:",
        options=display_options,
        key="chatbot_selector"
    )

    # Convert display name back to internal lowercase name
    selected_chatbot_name_internal = None
    if selected_chatbot_display != "Select a Chatbot":
        if selected_chatbot_display == "Philippine History":
            selected_chatbot_name_internal = "history"
        else:
            selected_chatbot_name_internal = selected_chatbot_display.lower()


    # Handle chatbot change
    if selected_chatbot_name_internal and st.session_state['current_chatbot_name'] != selected_chatbot_name_internal:
        st.session_state['current_chatbot_name'] = selected_chatbot_name_internal
        st.session_state['messages'] = [] # Clear chat history when switching chatbots
        st.toast(f"Switched to {selected_chatbot_display} Chatbot!")

        # Load or setup the RAG pipeline/Agent for the selected chatbot
        if selected_chatbot_name_internal not in st.session_state['rag_pipelines']: # Refer to new session state key
            with st.spinner(f"Setting up {selected_chatbot_display} chatbot... This may take a moment for the first time."):
                # setup_rag_pipeline now returns a single chain/agent, not a tuple
                pipeline = setup_rag_pipeline(selected_chatbot_name_internal)
                if pipeline: # Check if the pipeline object was successfully returned
                    st.session_state['rag_pipelines'][selected_chatbot_name_internal] = pipeline # Store the single object
                    st.success(f"{selected_chatbot_display} chatbot ready!")
                    st.session_state['messages'].append({"role": "assistant", "content": f"Hello! I am the {selected_chatbot_display} Chatbot. How can I assist you today?"})
                else:
                    st.error(f"Failed to set up {selected_chatbot_display} chatbot. Please check your document files in '{DOCUMENTS_DIRECTORY}' and API key.")
                    st.session_state['current_chatbot_name'] = None # Reset if setup fails
        else:
            # If already loaded, just set the initial message
            st.session_state['messages'].append({"role": "assistant", "content": f"Welcome back! I am the {selected_chatbot_display} Chatbot. What's your next question?"})

    # If no chatbot is selected, show an instruction in the sidebar
    if not st.session_state['current_chatbot_name']:
        st.markdown("---")
        st.info("Please select a chatbot from the dropdown above to begin interacting.")

# --- Main Chat Interface ---
if st.session_state['current_chatbot_name']:
    st.subheader(f"Chat with the {st.session_state['current_chatbot_name'].capitalize()} Chatbot")

    # Retrieve the active pipeline (which could be a RAG chain or an AgentExecutor)
    # This will be the single object returned by setup_rag_pipeline
    current_pipeline = st.session_state['rag_pipelines'].get(st.session_state['current_chatbot_name'])

    if current_pipeline: # Ensure the pipeline object exists
        # Display chat messages from history on app rerun
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input from user
        if prompt := st.chat_input(f"Ask your {st.session_state['current_chatbot_name'].capitalize()} chatbot a question..."):
            st.session_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        ai_response_content = "I couldn't get a response." # Default message
                        source_docs_display = [] # Initialize a list to hold source docs for display
                        was_rag_used = False # Initialize the flag (will be true if a RAG chain's context is available)

                        # Invoke the pipeline. The AgentExecutor (for inventory) will handle routing internally.
                        # For other chatbots, current_pipeline is simply the rag_chain.
                        response_obj = current_pipeline.invoke({"input": prompt})

                        # --- Handling different response formats (AgentExecutor vs. RAG Chain) ---
                        if isinstance(response_obj, dict):
                            # AgentExecutor's final answer is typically in 'output'
                            if 'output' in response_obj:
                                ai_response_content = response_obj['output']
                            # RAG chain's answer is typically in 'answer'
                            elif 'answer' in response_obj:
                                ai_response_content = response_obj['answer']
                                source_docs_display = response_obj.get("context", []) # Source docs for RAG
                                was_rag_used = True
                            else: # Fallback for unexpected dict structure
                                ai_response_content = str(response_obj) # Convert dict to string for display
                                print(f"DEBUG: Unexpected dictionary response structure: {response_obj}")
                        elif isinstance(response_obj, str): # Direct string response (less common with current LangChain versions)
                            ai_response_content = response_obj
                            print(f"DEBUG: Direct string response received: {response_obj}")
                        else:
                            ai_response_content = "Unexpected response format from the chatbot."
                            print(f"DEBUG: Unexpected response type: {type(response_obj)} - Value: {response_obj}")


                        # --- DEBUG PRINTS START ---
                        print(f"\n--- DEBUG: Final Response Info for '{st.session_state['current_chatbot_name'].capitalize()}' ---")
                        print(f"DEBUG: Answer: {ai_response_content[:100]}...") # Print first 100 chars of answer
                        print(f"DEBUG: was_rag_used: {was_rag_used}") # True if RAG chain was used AND it returned context

                        if was_rag_used and source_docs_display:
                            print(f"DEBUG: Number of Source Docs in source_docs_display: {len(source_docs_display)}")
                            print(f"DEBUG: First Source Doc Metadata: {source_docs_display[0].metadata}")
                        else:
                            print("DEBUG: No RAG context available (either not RAG, or no relevant docs found).")
                        print("----------------------------------------------------------------")
                        # --- DEBUG PRINTS END ---

                        # --- DISPLAY THE AI ANSWER ---
                        st.markdown(f"**Answer:** {ai_response_content}")
                        st.session_state['messages'].append({"role": "assistant", "content": ai_response_content})

                        # --- CONSOLIDATED SOURCE DOCUMENT DISPLAY (UNCOMMENT TO SHOW) ---
                        # If you want to show source documents again, uncomment this entire block.
                        # For now, it's suppressed to provide a cleaner answer-only output.
                        # if was_rag_used and source_docs_display:
                        #     st.markdown("---") # Separator for clarity
                        #     st.markdown("### **Retrieved Source Documents for RAG:**") 
                        #     for i, doc in enumerate(source_docs_display):
                        #         st.text(f"--- Document {i+1} ---")
                        #         st.code(doc.page_content, language="json" if ".json" in doc.metadata.get('source', '') else ("csv" if ".csv" in doc.metadata.get('source', '') else "text"))
                        # elif was_rag_used and not source_docs_display:
                        #      st.info(f"No relevant source documents were retrieved by {st.session_state['current_chatbot_name'].capitalize()} RAG for this query.")
                        # --- END CONSOLIDATED SOURCE DOCUMENT DISPLAY ---

                    except Exception as e:
                        error_message = f"An error occurred: {e}. Please try again. Ensure your OpenAI API Key is valid and that the necessary documents are in the '{DOCUMENTS_DIRECTORY}' folder."
                        st.error(error_message)
                        print(f"ERROR: Exception during query processing: {e}") # This will print to terminal
                        import traceback; traceback.print_exc() # Temporarily uncomment this for a full stack trace if an unexpected error occurs
                        st.session_state['messages'].append({"role": "assistant", "content": error_message})
    else: # This 'else' block executes if current_pipeline is None (meaning setup failed)
        st.warning("Failed to load chatbot components. Please try re-selecting the chatbot or check your setup.")
else: # This 'else' block executes if no chatbot is currently selected from the sidebar
    st.info("Choose a chatbot from the sidebar to begin!")

# --- FOOTER HTML (This remains at the very end of your file) ---
st.html(
    """
    <div class="footer">
        AI Engineer: Siegfried Manuel Eata.<br>
        Need help implementing intelligent chatbots and AI agents for your business or personal projects?<br>
        Please contact: <a href='mailto:manueleata@gmail.com' style='color: gray;'>manueleata@gmail.com</a> | 📞 <a href='tel:+639625609145' style='color: gray;'>+63 962 560 9145</a>
    </div>
    """
)