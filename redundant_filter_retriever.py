from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever

# This is a custom LangChain retriever that that accepts an embedding and a vectorstore (Chroma) as arguments.
# It is used to filter out redundant documents from the results of the retrieval step
# to prevent semantic similar documents from being retrieved.
class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        emb = self.embeddings.embed_query(query)

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8
        )

    async def aget_relevant_documents(self):
        return []