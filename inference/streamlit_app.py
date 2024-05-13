import os

import pandas as pd

import streamlit as st
from streamlit_searchbox import st_searchbox

from modules.streamlit_utils import read_df, search_name, get_instr_url


# Setup paths
HOME = os.getcwd()

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')
RCETH_CSV_PATH = os.path.join(DATA_DIR, 'rceth.csv')

# Read csv with instructions
df = read_df(rceth_csv_path=RCETH_CSV_PATH)

# App title
st.set_page_config(page_title="Medical instruction question app",
                   page_icon="💊", layout="wide")

# Change width of sidebar
st.markdown(
    """
<style>
[data-testid="stSidebar"][aria-expanded="true"]{
width: 500px;
max-width: 768px;
}
</style>
""",
    unsafe_allow_html=True)


def search(name):
    return search_name(df=df,
                       name=name)


# Sidebar
with st.sidebar:
    # Set title of sidebar
    st.title('Chat with Medical Instruction')
    # Change width of sidebar
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
    width: 500px;
    max-width: 768px;
    }
    </style>
    """,
        unsafe_allow_html=True)

    # Search
    st.subheader('Лекарственный препарат')
    text_search = st_searchbox(search_function=search,
                               placeholder="Введите название ЛП")

    if text_search:
        instr_urls = df.loc[df["full_name"] ==
                            text_search, "link_of_instruction"].values[0]
        if instr_urls:
            instr_url = get_instr_url(instr_urls)
            print(instr_url)

        else:
            instr_url = None

    st.markdown(
        '📖 Все инструкции взяты с сайта [«Государственный реестр лекарственных средств Республики Беларусь»](https://rceth.by/Refbank/).')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Что тебя интересует в данной инструкции?"}]

# Display or clear chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Что тебя интересует в данной инструкции?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response


def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = "Output from LLM"
    # output = replicate.run(llm,
    #                        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
    #                               "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output


# User-provided prompt
if prompt := st.chat_input():  # (disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
