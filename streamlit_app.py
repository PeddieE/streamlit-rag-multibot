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
st.info("The knowledge bases for each chatbot are built on your local documents.")

# --- Session State Initialization ---
# This ensures that the RAG chains and chat history persist across reruns
if 'rag_chains' not in st.session_state:
    st.session_state['rag_chains'] = {}
if 'current_chatbot_name' not in st.session_state:
    st.session_state['current_chatbot_name'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


# --- Sidebar for Chatbot Selection ---
with st.sidebar:
    st.header("Chatbot Selection")

    available_chatbots = sorted(list(CHATBOT_DOCUMENTS.keys())) # Sort for consistent display

    # Add a "None" or "Select a Chatbot" option
    # UPDATED: Changed "History" to "Philippine History" in display
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
        # UPDATED: Handle "Philippine History" to "history" mapping for internal use
        if selected_chatbot_display == "Philippine History":
            selected_chatbot_name_internal = "history"
        else:
            selected_chatbot_name_internal = selected_chatbot_display.lower()


    # Handle chatbot change
    if selected_chatbot_name_internal and st.session_state['current_chatbot_name'] != selected_chatbot_name_internal:
        st.session_state['current_chatbot_name'] = selected_chatbot_name_internal
        st.session_state['messages'] = [] # Clear chat history when switching chatbots
        st.toast(f"Switched to {selected_chatbot_display} Chatbot!")

        # Load or setup the RAG pipeline for the selected chatbot
        if selected_chatbot_name_internal not in st.session_state['rag_chains']:
            with st.spinner(f"Setting up {selected_chatbot_display} chatbot... This may take a moment for the first time."):
                rag_chain = setup_rag_pipeline(selected_chatbot_name_internal)
                if rag_chain:
                    st.session_state['rag_chains'][selected_chatbot_name_internal] = rag_chain
                    st.success(f"{selected_chatbot_display} chatbot ready!")
                    st.session_state['messages'].append({"role": "assistant", "content": f"Hello! I am the {selected_chatbot_display} Chatbot. How can I assist you today?"})
                else:
                    st.error(f"Failed to set up {selected_chatbot_display} chatbot. Please check your document files in '{DOCUMENTS_DIRECTORY}'.")
                    st.session_state['current_chatbot_name'] = None # Reset if setup fails
        else:
            # If already loaded, just set the initial message
            st.session_state['messages'].append({"role": "assistant", "content": f"Welcome back! I am the {selected_chatbot_display} Chatbot. What's your next question?"})

    # If no chatbot is selected, show an instruction in the sidebar
    if not st.session_state['current_chatbot_name']:
        st.markdown("---")
        st.info("Please select a chatbot from the dropdown above to begin interacting.")

    # Optional: Display API Key input for user (if you want to allow users to provide their own)
    # api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    # if api_key:
    #    os.environ["OPENAI_API_KEY"] = api_key
    #    st.success("API Key set!")
    # else:
    #    st.warning("Please enter your OpenAI API Key.")

# --- Main Chat Interface ---
if st.session_state['current_chatbot_name']:
    st.subheader(f"Chat with the {st.session_state['current_chatbot_name'].capitalize()} Chatbot")

    current_rag_chain = st.session_state['rag_chains'].get(st.session_state['current_chatbot_name'])

    if current_rag_chain:
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
                        response = current_rag_chain.invoke({"input": prompt})
                        ai_response = response["answer"]
                        st.markdown(ai_response)
                        st.session_state['messages'].append({"role": "assistant", "content": ai_response})

                        # Optional: Display retrieved context in an expander
                        # with st.expander("Show Retrieved Context"):
                        #    for i, doc in enumerate(response["context"]):
                        #        source_info = Path(doc.metadata.get('source', 'N/A')).name
                        #        page_info = doc.metadata.get('page', 'N/A')
                        #        row_info = doc.metadata.get('row', 'N/A')
                        #        location_info = f"Source: {source_info}"
                        #        if page_info != 'N/A':
                        #            location_info += f", Page: {page_info}"
                        #        if row_info != 'N/A':
                        #            location_info += f", Row: {row_info}"
                        #        st.write(f"**Chunk {i+1} ({location_info}):**")
                        #        st.text(doc.page_content[:500] + "...") # Use st.text for preformatted text

                    except Exception as e:
                        error_message = f"An error occurred: {e}. Please try again. Ensure your OpenAI API Key is valid and that the necessary documents are in the '{DOCUMENTS_DIRECTORY}' folder."
                        st.error(error_message)
                        st.session_state['messages'].append({"role": "assistant", "content": error_message})
    else: # This 'else' block executes if current_rag_chain is None (meaning setup failed)
        st.warning("Please select a chatbot from the sidebar to start.")
else: # This 'else' block executes if no chatbot is currently selected from the sidebar
    st.info("Choose a chatbot from the sidebar to begin!")

# --- FOOTER HTML (This remains at the very end of your file) ---
# This uses Streamlit's st.html to inject raw HTML for the fixed footer.
st.html(
    """
    <div class="footer">
        Developer: Siegfried Manuel Eata.<br>
        Need help implementing intelligent chatbots for your business or personal projects?<br>
        Please contact: <a href='mailto:manueleata@gmail.com' style='color: gray;'>manueleata@gmail.com</a> | 📞 <a href='tel:+639625609145' style='color: gray;'>+63 962 560 9145</a>
    </div>
    """
)
# --- END FOOTER HTML ---