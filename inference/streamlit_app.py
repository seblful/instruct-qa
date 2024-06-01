import os

import pandas as pd

import streamlit as st
from streamlit_searchbox import st_searchbox

from marker.models import load_all_models

from modules.instructors import Instruction
from modules.detectors import ImageProcessor
from modules.llm_qa import VectorSearcher, RAGAgent


# Setup paths
HOME = os.getcwd()

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')
CLEAN_INSTR_DIR = os.path.join(DATA_DIR, "clean-instructions")
EXTR_INSTR_DIR = os.path.join(DATA_DIR, "extracted-instructions")
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


# Read csv with instructions
@st.cache_data(show_spinner=True)
def read_df(rceth_csv_path):
    print("LOAD DF")
    df = pd.read_csv(rceth_csv_path, encoding='windows-1251')
    df["full_name"] = df["trade_name"] + " " + \
        df["dosage_form"] + " " + df["manufacturer"]

    return df


# Load surya models
@st.cache_resource()
def load_models():
    surya_model_list = load_all_models()

    return surya_model_list


df = read_df(rceth_csv_path=RCETH_CSV_PATH)
surya_model_list = load_models()

# Load processor for image processing
if "image_processor" not in st.session_state:
    st.session_state["image_processor"] = ImageProcessor(yolo_stamp_det_model_path=YOLO_STAMP_DET_MODEL_PATH,
                                                         segformer_la_model_path=SEGFORMER_LA_MODEL_PATH,
                                                         segformer_la_config_path=SEGFORMER_LA_CONFIG_PATH,
                                                         surya_model_list=surya_model_list)

# Load vectorsearcher and RAG agent
if "vector_searcher" not in st.session_state:
    st.session_state["vector_searcher"] = VectorSearcher(db_name="faiss",
                                                         yandex_api_key=YANDEX_API_KEY,
                                                         yandex_folder_id=YANDEX_FOLDER_ID,
                                                         chunk_size=1000,
                                                         chunk_overlap=200,
                                                         opensearch_url=OPENSEARCH_HOST,
                                                         opensearch_login=OPENSEARCH_LOGIN,
                                                         opensearch_password=OPENSEARCH_PASSWORD)
if "rag_agent" not in st.session_state:
    st.session_state["rag_agent"] = RAGAgent(yandex_api_key=YANDEX_API_KEY,
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
                        clean_instr_dir,
                        extr_instr_dir,
                        _image_processor,
                        _vector_searcher):
    # Get instruction url and create Instruction instance
    instr_url = get_instr_url(instr_urls)
    instruction = Instruction(instr_dir=instr_dir,
                              clean_instr_dir=clean_instr_dir,
                              extr_instr_dir=extr_instr_dir,
                              pdf_path_or_url=instr_url)

    # Extract text from instruction and save as markdown
    instruction.extract_text(image_processor=_image_processor)
    full_md_path = os.path.join(
        instruction.extr_instr_dir, instruction.md_path)

    # Create vectorsearch
    vectorsearch = _vector_searcher.create_vectorsearch(md_path=full_md_path)

    return vectorsearch


def clear_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Что Вас интересует в данной инструкции?"}]


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
                                                   clean_instr_dir=CLEAN_INSTR_DIR,
                                                   extr_instr_dir=EXTR_INSTR_DIR,
                                                   _image_processor=st.session_state["image_processor"],
                                                   _vector_searcher=st.session_state["vector_searcher"])

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
            response = st.session_state["rag_agent"].get_answer(question=prompt,
                                                                vectorsearch=vectorsearch)
        st.markdown(response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
