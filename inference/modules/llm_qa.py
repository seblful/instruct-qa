from opensearchpy import OpenSearch


from langchain_community.vectorstores import OpenSearchVectorSearch, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from yandex_chain import YandexEmbeddings, YandexLLM


class LLMQA:
    pass


class VectorSearcher:
    def __init__(self,
                 db_name):
        assert db_name in [
            "opensearch", "faiss"], "Database should be in ['opensearch', 'faiss']."
        self.db_name = db_name

    def create_vectorsearch(self,
                            documents,
                            embeddings,
                            opensearch_url='localhost',
                            opensearch_login='admin',
                            opensearch_password='admin'):
        if self.db_name == "faiss":
            vectorsearch = FAISS.from_documents(documents, embeddings)

        elif self.db_name == "opensearch":
            vectorsearch = OpenSearchVectorSearch.from_documents(
                documents,
                embeddings,
                opensearch_url=opensearch_url,
                http_auth=(opensearch_login, opensearch_password),
                use_ssl=True,
                verify_certs=False,
                engine='lucene')

        return vectorsearch


class RAGAgent:
    def __init__(self,
                 yandex_api_key,
                 yandex_folder_id,
                 chunk_size=1000,
                 chunk_overlap=200,
                 temperature=0.8,
                 max_tokens=3000):
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)

        # Embeddings and llm
        self.embeddings = YandexEmbeddings(
            folder_id=yandex_folder_id, api_key=yandex_api_key)
        self.llm = YandexLLM(api_key=yandex_api_key,
                             folder_id=yandex_folder_id,
                             temperature=temperature,
                             max_tokens=max_tokens,
                             use_lite=False)

        # Template and prompt
        self.template = """Ты компетентный ИИ-помощник, разбирающийся в лекарственных препаратах и медицине. 
        Твоя задача дать полные ответы на вопросы на русском языке в рамках предоставленного ниже текста (контекста).
        Отвечай точно в рамках предоставленного текста, даже если тебя просят придумать.
        Eсли знаешь больше, чем указано в тексте, а внутри текста ответа нет, отвечай:
        "Я могу давать ответы только по тематике загруженных документов. Мне не удалось найти в документах ответ на ваш вопрос."

        Контекст: {context}

        Вопрос: {question}
        
        Твой ответ:
        """
        self.prompt = PromptTemplate.from_template(self.template)

        # Retriever
        self.vector_searcher = VectorSearcher(db_name="faiss")

    def get_answer(self,
                   text,
                   question):
        # Get documents
        documents = self.text_splitter.create_documents([text])

        # Get vectorsearch
        vectorsearch = self.vector_searcher.create_vectorsearch(documents=documents,
                                                                embeddings=self.embeddings)
        # RetrievalQA
        qa = RetrievalQA.from_chain_type(
            self.llm,
            retriever=vectorsearch.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt})

        # Get results and answer
        results = qa.invoke({'query': question})
        answer = results['result']

        return answer
