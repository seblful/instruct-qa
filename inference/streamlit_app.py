import os

import pandas as pd

import streamlit as st
from streamlit_searchbox import st_searchbox


# Setup paths
HOME = os.getcwd()

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')
RCETH_CSV_PATH = os.path.join(DATA_DIR, 'rceth.csv')

# Read csv with instructions
df = pd.read_csv(RCETH_CSV_PATH, encoding='windows-1251')
instr_names = df.loc[:, "trade_name"].to_list()

# App title
st.set_page_config(page_title="Medical instruction question app",
                   page_icon="💊", layout="wide")


def search_name(name):
    contain_names = df['trade_name'].str.contains(name.lower(), case=False)
    med_df = df.loc[(contain_names) & (
        df["dosage_form"] != "субстанция"), ["trade_name", "dosage_form", "manufacturer"]]

    # Concatenate name, dosage_form and manufacturer
    med_df["full_name"] = med_df["trade_name"] + " " + \
        med_df["dosage_form"] + " " + med_df["manufacturer"]
    names = med_df["full_name"].to_list()

    return names


# Change width of sidebar
st.markdown(
    """
<style>
section[data-testid="stSidebar"] {
width: 500px !important;
</style>
""",
    unsafe_allow_html=True
)


# Sidebar
with st.sidebar:
    st.title('Chat with Medical Instruction')

    # Search
    st.subheader('Лекарственный препарат')
    text_search = st_searchbox(search_function=search_name,
                               placeholder="Введите название ЛП")

    if text_search:
        pass

    temperature = st.sidebar.slider(
        'temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01,
                              max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider(
        'max_length', min_value=64, max_value=4096, value=512, step=8)

    st.markdown(
        '📖 Все инструкции взяты с сайта [«Государственный реестр лекарственных средств Республики Беларусь»](https://rceth.by/Refbank/).')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How may I assist you today?"}]


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
