import os

from modules.instructors import Instruction
from modules.detectors import InstructionProcessor
from modules.llm_qa import VectorSearcher, RAGAgent


# Setup paths
HOME = os.getcwd()

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')

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


def main():
    pdf_path = os.path.join(INSTR_DIR, '21_07_3165_i.pdf')
    pdf_url = "https://rceth.by//NDfiles/instr/7401_05_10_15_18_20_s.pdf"
    instruction = Instruction(instr_dir=INSTR_DIR,
                              pdf_url=pdf_url)  # , pdf_path=pdf_path)

    # Create InstructionProcessor instance
    instr_processor = InstructionProcessor(instr_dir=INSTR_DIR,
                                           yolo_stamp_det_model_path=YOLO_STAMP_DET_MODEL_PATH,
                                           segformer_la_model_path=SEGFORMER_LA_MODEL_PATH,
                                           segformer_la_config_path=SEGFORMER_LA_CONFIG_PATH)

    # Extract tect from instruction
    text = instr_processor.extract_text(instruction=instruction)

    # with open("texts.txt", 'r', encoding="windows-1251") as file:
    #     text = file.read()

    # Get answer using RAG LLM
    question = "Какая область применения?"
    vector_searcher = VectorSearcher(db_name="faiss",
                                     yandex_api_key=YANDEX_API_KEY,
                                     yandex_folder_id=YANDEX_FOLDER_ID,
                                     chunk_size=1000,
                                     chunk_overlap=200,
                                     opensearch_url=OPENSEARCH_HOST,
                                     opensearch_login=OPENSEARCH_LOGIN,
                                     opensearch_password=OPENSEARCH_PASSWORD)
    vectorsearch = vector_searcher.create_vectorsearch(text=text)

    rag_agent = RAGAgent(yandex_api_key=YANDEX_API_KEY,
                         yandex_folder_id=YANDEX_FOLDER_ID,
                         temperature=0.8,
                         max_tokens=3000)

    answer = rag_agent.get_answer(question=question,
                                  vectorsearch=vectorsearch)

    print(answer)


if __name__ == '__main__':
    main()
