import streamlit as st
import time
import os 
# Prev model - from logic__single_model_development.model_embeddings_prod import *
# Prev model - from logic__conversational_model_development.model_embeddings_prod_with_memory import *
from logic__conversational_model_development.model_embeddings_prod_with_memory_chained_prompt import chained_prompt
from hugchat.login import Login
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Function should only run once when called - https://docs.streamlit.io/library/api-reference/performance/st.cache
# https://discuss.streamlit.io/t/avoid-rerunning-some-code/1313
@st.cache_data
def get_candidates():
    # candidates = {'Chris Christie': 'chris_christie',
    #              'Dean Phillips': 'dean_phillips',
    directory = 'embeddings'
    candidates = {}
 
    for filename in os.listdir(directory):
        parsed_filename = filename.split("faiss_index_")[1]
        formatted_name = parsed_filename.replace('_', ' ').title()
        candidates[formatted_name] = parsed_filename
    
    return candidates

# if a new candidate is chosen, reset the memory and messages
def reset_memory():
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    print('^^ Memory Reset ^^')
    return

# Useful resources: 
# - https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# - https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/
# - https://www.youtube.com/watch?v=_j7JEDWuqLE&ab_channel=AIJason
def main():

    # Initialization setup - should only run once, unless new candidates are added
    st.session_state.candidates = get_candidates()

    st.title("All In Podcast - Presidential Candidate Q&A")

    with st.sidebar:
        st.title('💬 Candidate Selection')

        option = st.selectbox(
                'Who do you want to learn more about?',
                get_candidates().keys(),
                index=None,
                placeholder="Select a candidate...",
                on_change=reset_memory
                )
        
        st.write('You selected:', option)
        
        if not (option):
            st.warning('Please select a candidate!', icon='⚠️')
        else:
            st.success('Proceed to using the chat functionality!', icon='👉')
            
        st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

    # Initialize chat history and memory
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferWindowMemory(k=3, return_messages=True)


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me a question!"):

        if not(option):
            st.warning('Please select a candidate using the dropdown', icon='⚠️')
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()            
                response = chained_prompt(prompt, 
                                        st.session_state.memory, 
                                        st.session_state.candidates[option], 
                                        False)['answer']
                st.session_state.memory.save_context({"input": prompt}, {"output": response})
                # response = method_3(prompt, st.session_state.memory, False)['answer']
                # st.session_state.memory.save_context({"input": prompt}, {"output": response})
                
                print("############")
                print(st.session_state.memory.load_memory_variables({}))
                print(st.session_state.messages)
                
                full_response = ""

                # Simulate stream of response with milliseconds delay
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()


# To run from console -> streamlit run app.py