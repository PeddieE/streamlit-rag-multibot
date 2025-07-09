if current_pipeline_tuple: # Ensure the tuple exists
    current_rag_chain, current_inventory_agent = current_pipeline_tuple
    # Now current_rag_chain is your chain object and current_inventory_agent is your agent object (or None)

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
                    ai_response = "I couldn't get a response." # Default message
                    source_docs_display = [] # Initialize a list to hold source docs for display

                    # --- ROUTING LOGIC BASED ON CHATBOT NAME ---
                    if st.session_state['current_chatbot_name'] == "inventory" and current_inventory_agent:
                        # Routing for inventory: if it looks like a numerical/listing query, use the agent
                        # Updated keywords now include "show all", "list all", "all products", "display all"
                        if any(word in prompt.lower() for word in ["what is the total", "how many", "count", "sum", "average", "value of", "cheapest", "most expensive", "stock level", "highest", "lowest", "max", "min", "list all", "show all", "all products", "display all"]):
                            print("DEBUG: Routing to Inventory Agent.")
                            # --- FIX: Removed temporary debugging code and duplicate invoke ---
                            agent_response = current_inventory_agent.invoke({"input": prompt})
                            ai_response = agent_response.get('output', str(agent_response))
                            # Agents generally don't return "source_documents" in the same way RAG chains do.
                            # So, source_docs_display remains empty for agent responses here.
                        else: # Routing to Inventory RAG Chain for descriptive info
                            print("DEBUG: Routing to Inventory RAG Chain.")
                            response_rag = current_rag_chain.invoke({"input": prompt})
                            ai_response = response_rag["answer"]
                            source_docs_display = response_rag.get("source_documents", []) # Populate source_docs_display for RAG

                    else: # For all other chatbots (medical, history, finance, legal) or if inventory agent isn't available
                        print(f"DEBUG: Routing to RAG Chain for {st.session_state['current_chatbot_name']}.")
                        response = current_rag_chain.invoke({"input": prompt})
                        ai_response = response["answer"]
                        source_docs_display = response.get("source_documents", []) # Populate source_docs_display for RAG

                    st.markdown(ai_response)
                    st.session_state['messages'].append({"role": "assistant", "content": ai_response})

                    # --- Consolidated Source Document Display (applies to RAG chains only) ---
                    if source_docs_display:
                        st.markdown("---") # Separator for clarity
                        st.markdown(f"**Retrieved Source Documents for {st.session_state['current_chatbot_name'].capitalize()} RAG:**")
                        for i, doc in enumerate(source_docs_display):
                            st.text(f"--- Document {i+1} ---")
                            st.code(doc.page_content, language="json" if doc.metadata.get('file_type') == 'json' else "text")
                            # Optional: Display metadata
                            # st.markdown("Metadata:")
                            # st.json(doc.metadata)
                    elif 'current_chatbot_name' in st.session_state and not source_docs_display:
                        # This condition ensures the message only appears for RAG chains that were actually invoked
                        # and did not find documents. It will not show for agent responses.
                        # The specific message depends on which RAG chain was active.
                        st.info(f"No relevant source documents were retrieved by {st.session_state['current_chatbot_name'].capitalize()} RAG for this query.")
                    # --- END Consolidated Source Document Display ---

                except Exception as e:
                    error_message = f"An error occurred: {e}. Please try again. Ensure your OpenAI API Key is valid and that the necessary documents are in the '{DOCUMENTS_DIRECTORY}' folder."
                    st.error(error_message)
                    print(f"ERROR: Exception during query processing: {e}") # This will print to terminal
                    import traceback; traceback.print_exc() # Temporarily uncomment this for a full stack trace if an unexpected error occurs
                    st.session_state['messages'].append({"role": "assistant", "content": error_message})
else: # This 'else' block executes if current_pipeline_tuple is None (meaning setup failed)
    st.warning("Failed to load chatbot components. Please try re-selecting the chatbot or check your setup.")