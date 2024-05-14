import os

import pandas as pd

import streamlit as st
from streamlit_searchbox import st_searchbox

from modules.instructors import Instruction
from modules.detectors import InstructionProcessor
from modules.llm_qa import VectorSearcher, RAGAgent


# Setup paths
HOME = os.getcwd()

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')
RCETH_CSV_PATH = os.path.join(DATA_DIR, 'rceth.csv')

# Models
MODELS_DIR = os.path.join(HOME, "models")
YOLO_STAMP_DET_MODEL_PATH = os.path.join(MODELS_DIR, "yolo_stamp_det.pt")
SEGFORMER_LA_MODEL_PATH = os.path.join(MODELS_DIR, "segformer_la.ckpt")
SEGFORMER_LA_CONFIG_PATH = os.path.join(MODELS_DIR, "segformer_la_config.json")

# Envs
OPENSEARCH_LOGIN = os.getenv('OPENSEARCH_LOGIN')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD')

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

# Other
OPENSEARCH_HOST = 'localhost'
OPENSEARCH_PORT = 9200

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Read csv with instructions
@st.cache_data(show_spinner=False)
def read_df(rceth_csv_path):
    df = pd.read_csv(rceth_csv_path, encoding='windows-1251')
    df["full_name"] = df["trade_name"] + " " + \
        df["dosage_form"] + " " + df["manufacturer"]

    return df


df = read_df(rceth_csv_path=RCETH_CSV_PATH)

# Load processor for text extraction
instr_processor = InstructionProcessor(instr_dir=INSTR_DIR,
                                       yolo_stamp_det_model_path=YOLO_STAMP_DET_MODEL_PATH,
                                       segformer_la_model_path=SEGFORMER_LA_MODEL_PATH,
                                       segformer_la_config_path=SEGFORMER_LA_CONFIG_PATH)

# Load vectorsearcher and RAG agent
vector_searcher = VectorSearcher(db_name="faiss",
                                 yandex_api_key=YANDEX_API_KEY,
                                 yandex_folder_id=YANDEX_FOLDER_ID,
                                 chunk_size=1000,
                                 chunk_overlap=200,
                                 opensearch_url=OPENSEARCH_HOST,
                                 opensearch_login=OPENSEARCH_LOGIN,
                                 opensearch_password=OPENSEARCH_PASSWORD)

rag_agent = RAGAgent(yandex_api_key=YANDEX_API_KEY,
                     yandex_folder_id=YANDEX_FOLDER_ID,
                     temperature=0.8,
                     max_tokens=3000)


def search_name(name):
    # Find name in dataframe and create subdataframe
    contain_names = df['trade_name'].str.contains(
        name.lower(), case=False)
    med_series = df.loc[(contain_names) & (
        df["dosage_form"] != "субстанция"), "full_name"]
    med_series = med_series.sort_values()

    # Sort values and convert to list
    names = med_series.to_list()

    return names


def get_instr_url(instr_urls):
    # Split string and strip each instruction
    instr_urls = instr_urls.split(",")
    instr_urls = [instr.strip() for instr in instr_urls]

    # Sort instruction by last letter in basename
    instr_urls.sort(key=lambda x: os.path.splitext(
        os.path.basename(x))[0][-1])

    return instr_urls[-1]


@st.cache_data(show_spinner=False)
def process_instruction(instr_urls,
                        instr_dir,
                        _instr_processor,
                        _vector_searcher):
    # Get instruction url and create Instruction instance
    instr_url = get_instr_url(instr_urls)
    instruction = Instruction(instr_dir=instr_dir,
                              pdf_url=instr_url)

    # Extract text from instruction
    text = _instr_processor.extract_text(instruction=instruction)

    # Create vectorsearch
    vectorsearch = _vector_searcher.create_vectorsearch(text=text)

    return vectorsearch


def clear_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Что Вас интересует в данной инструкции?"}]


# App title
st.set_page_config(page_title="Chat with Medical Instruction",
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
""", unsafe_allow_html=True)


if "last_text_search" not in st.session_state.keys():
    st.session_state["last_text_search"] = ""

# Sidebar
with st.sidebar:
    # Set title of sidebar
    st.title("Chat with Medical Instruction")

    # Search
    st.subheader('Лекарственный препарат')
    text_search = st_searchbox(search_function=search_name,
                               placeholder="Введите название ЛП")

    if text_search:
        if st.session_state["last_text_search"] != text_search:
            clear_chat_history()
            st.session_state["last_text_search"] = text_search

        # Get instructions urls
        instr_urls = df.loc[df["full_name"] ==
                            text_search, "link_of_instruction"].values[0]
        if instr_urls:
            with st.spinner(text="Обработка инструкции..."):
                # Process instructions urls, extract text and create vectorsearch
                vectorsearch = process_instruction(instr_urls=instr_urls,
                                                   instr_dir=INSTR_DIR,
                                                   _instr_processor=instr_processor,
                                                   _vector_searcher=vector_searcher)

        else:
            vectorsearch = None

    # Clear chat messages
    st.sidebar.button('Очистить историю чата', on_click=clear_chat_history)

    # Markdown
    st.markdown(
        '📖 Все инструкции взяты с сайта [«Государственный реестр лекарственных средств Республики Беларусь»](https://rceth.by/Refbank/).')

# Store UI generated responses
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Что Вас интересует в данной инструкции?"}]

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(placeholder="Ваш вопрос", disabled=not text_search):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Генерация ответа..."):
            response = rag_agent.get_answer(question=prompt,
                                            vectorsearch=vectorsearch)
        placeholder = st.empty()
        placeholder.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
