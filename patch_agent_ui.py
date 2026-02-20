import re

file_agent = 'pages/7_🎯_AI_Agent.py'

with open(file_agent, 'r', encoding='utf-8') as f:
    text_agent = f.read()

# The UI currently stops displaying Insights after `st.markdown("---")` near line 163 (inside the `if 'ai_insights'` block)
# Actually, let's inject a new section right at the end of the `if 'ai_insights' in st.session_state:` block.

new_chat_section = '''
    # NEW INTERACTIVE CHAT SECTION
    st.markdown("---")
    st.markdown("### 💬 Ask the AI Agent")
    st.info("The Agent has analyzed the dataset. Now you can ask it to perform specific tasks, generate custom code, or answer questions about your data!")
    
    # Initialize agent chat history if not exists
    if 'agent_chat_history' not in st.session_state:
        st.session_state.agent_chat_history = []
        
    # Display chat history
    for msg in st.session_state.agent_chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("code"):
                with st.expander("Show Python Code", expanded=False):
                    st.code(msg["code"], language="python")
            if msg.get("data") is not None:
                if isinstance(msg["data"], pd.DataFrame):
                    st.dataframe(msg["data"])
                else:
                    st.write(msg["data"])

    # Chat input
    if prompt := st.chat_input("Ask the Agent to do something with your data... (e.g. 'Show the top 5 earners')"):
        # Add user message to state and display
        st.session_state.agent_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Agent is thinking and writing code..."):
                response = ai_chat.ask_question(df, prompt)
                
                if isinstance(response, dict):
                    st.markdown(response["answer"])
                    
                    if response.get("code"):
                        with st.expander("Show Executed Python Code", expanded=False):
                            st.code(response["code"], language="python")
                            
                    if response.get("data") is not None:
                        if isinstance(response["data"], pd.DataFrame):
                            st.dataframe(response["data"])
                        else:
                            st.write(response["data"])
                            
                    st.session_state.agent_chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "code": response.get("code"),
                        "data": response.get("data")
                    })
                else:
                    st.markdown(response)
                    st.session_state.agent_chat_history.append({"role": "assistant", "content": response})

else:
'''

if "### 💬 Ask the AI Agent" not in text_agent:
    # Replace the `else:` that corresponds to `if 'ai_insights' in st.session_state:`
    # We find the `else:` block around line 159
    text_agent = text_agent.replace('\nelse:\n    # Show placeholder', new_chat_section + '    # Show placeholder')
    with open(file_agent, 'w', encoding='utf-8') as f:
        f.write(text_agent)
    print("Injected Interactive Chat into AI Agent")
else:
    print("Interactive Chat already exists in AI Agent")
