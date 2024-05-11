from opensearchpy import OpenSearch
from langchain.vectorstores import FAISS, OpenSearchVectorSearch


class LLMQA:
    pass


class VectorStorer:
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
    pass
