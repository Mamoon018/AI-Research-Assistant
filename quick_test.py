
# Here we will create a tool to access the Supabase vector database to retrieve the data from it.

import asyncio
import langchain.embeddings
from langchain_openai import OpenAIEmbeddings
from src.ingestion.data_ingest import PDF_parser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from src.utils.model_initializer import supabase_client
from src.utils.settings import settings , get_api_key
from typing import Any
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from typing import List, Tuple, Optional, Dict, Any

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from typing import List, Tuple, Optional, Dict, Any

class CustomSupabaseVectorStore(SupabaseVectorStore):
    async def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 1,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector with relevance scores."""
        try:
            query = (
                self._client.table(self.table_name)
                .select("content, metadata")
                .order(
                    f"embedding <=> '{embedding}'::vector",
                    desc=False,
                )
                .limit(k)
            )
            if filter:
                for key, value in filter.items():
                    query = query.eq(key, value)
            res = await query.execute()  # Await the async query
            searches = res.data
            documents_with_scores = []
            for search in searches:
                metadata = (
                    search["metadata"] if search.get("metadata") is not None else {}
                )
                document = Document(page_content=search["content"], metadata=metadata)
                similarity = search.get("similarity", 0.0)
                documents_with_scores.append((document, similarity))
            return documents_with_scores
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}") from e

    async def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 1,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector."""
        try:
            result = await self.similarity_search_by_vector_with_relevance_scores(
                embedding, k=k, filter=filter, **kwargs
            )  # Await the async method
            documents = [doc for doc, _ in result]
            return documents
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search by vector: {e}") from e

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        try:
            embedding = self._embedding.embed_query(query)
            documents = await self.similarity_search_by_vector(
                embedding, k=k, filter=filter, **kwargs
            )  # Await the async method
            return documents
        except Exception as e:
            raise RuntimeError(f"Failed to perform async similarity search: {e}") from e

class database_queries():
    openai_embeddings = OpenAIEmbeddings()
    supabase_client_activated = supabase_client(asyncronous_supabase=True)
    vector_store = None
    documents: list[Any] = None
    no_of_doc_to_retrieve: int = 1


    async def initialize_vector_store(self):
        try:
            if self.documents:  # If file was uploaded and questions are asked about it.
                # Let's store the chunks in DB
                self.vector_store = CustomSupabaseVectorStore.from_documents(  # Use custom class
                    documents=self.documents,
                    embedding=self.openai_embeddings,
                    client=await self.supabase_client_activated,
                    table_name='documents'
                )
            
            elif not self.vector_store:  # When file was not provided
                self.vector_store = CustomSupabaseVectorStore(  # Use custom class
                    embedding=self.openai_embeddings,
                    client=await self.supabase_client_activated,
                    table_name='documents'
                )

        except Exception as e:
            raise RuntimeError(f'could not initialize the vectorstore due to runtime error {e}') from e   


    # Lets create a function for data retrieval 
    async def vector_embeddings_storage( self,doc_objects: list = None):
        """
        It takes the document objects created by PDF Parser to split into chunks then creates embeddings & store it in Supabase database. 

        **Args:**
        doc_objects (list): It is the list of the document objects. 

        **Returns:**
        It stores the embeddings in the database.

        **Raises:**
        Raises error if Supabase database is not accessible.
        """

        try: 
            # Lets create chunks of the list of objects created by PDFParser
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
            chunks = text_splitter.split_documents(doc_objects)
            self.documents = chunks

            # Initialize vectorstore to store data in Supabase database
            await self.initialize_vector_store()

        except Exception as e:
            raise RuntimeError(f'Vector embeddings could not store due to error {e}') from e 
        
    
    async def vector_data_retrieval(self, user_query: str) -> list:
        """
        It takes the user query, and embed it into vectors on a run-time. After that, retrieves the relevant content
        from the database based on given type of the search.

        **Args:**
        user_query (str): It is the question of the user related that we will be using to search content from the database.

        **Returns:**
        list (Any): It returns the list of the document objects retrieved from database that are related to the user query.
        
        **Raises:**
        It raises error if database is not accessible!

        """

        try:
            # initialize the vectorestore without chunks: Means just to connect to database for retrieval not to store chunks
            await self.initialize_vector_store()

            # lets retrieve the data using get_relevant_documents search
            retriever = self.vector_store.as_retriever(search_kwargs={"k": self.no_of_doc_to_retrieve})
            retrieved_data = await retriever.ainvoke(user_query)

            return retrieved_data

        except Exception as e:
            raise RuntimeError(f'data could not retrieved due to error {e}') from e


async def main():

    db = database_queries()
        
        # Example: Store documents
    file_address = 'src/ingestion/LangGraph.pdf'

    #doc_objects = await PDF_parser(file_path=file_address) # Your document objects
    #await db.vector_embeddings_storage()
        
        # Example: Retrieve data
    results = await db.vector_data_retrieval("What are spellings of AI?")
    print(results)

if __name__ == "__main__":
   asyncio.run(main())










    # we can use the as_retriever() to extract from DB.
    #print('embedding stored')
    

#async def main():
#    file_address = 'src/ingestion/LangGraph.pdf'
#    list_doc_objs = await PDF_parser(file_path=file_address)

#    await vector_embeddings(doc_objects=list_doc_objs)

#if __name__ == '__main__':
#    asyncio.run(main()) 