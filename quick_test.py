
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

class database_queries():
    openai_embeddings = OpenAIEmbeddings()
    supabase_client_activated = supabase_client(asyncronous_supabase=True)
    vector_store = None
    documents: list[Any] = None
    no_of_doc_to_retrieve: int = 1


    # We cannot await client at class level so, we have to do it within async initializer
    async def async_init(self):
        # Await the async supabase client correctly
        self.supabase_client_activated = await supabase_client(asyncronous_supabase=True)


    async def initialize_vector_store(self):
        try:
            if self.documents:  # If file was uploaded and questions are asked about it.
                # Let's store the chunks in DB
                self.vector_store = SupabaseVectorStore.from_documents(  # Use custom class
                    documents=self.documents,
                    embedding=self.openai_embeddings,
                    client= self.supabase_client_activated,
                    table_name='documents'
                )
            
            elif not self.vector_store:  # When file was not provided
                self.vector_store = SupabaseVectorStore(  # Use custom class
                    embedding=self.openai_embeddings,
                    client= self.supabase_client_activated,
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

            # lets get the vectors for user query
            query_embeddings = await self.openai_embeddings.aembed_query(user_query)

            # lets get the client that will be used to call rpc
            supabase_client = self.supabase_client_activated

            # Now we need to call the rpc (stored procedure created in the Supabase) that will receive the query embeddings
            # as input and executes vector similarity search for it and then returns the output

            rpc_response = await supabase_client.rpc("similarity_search_fn", {
                "query_vectors": query_embeddings,
                "k": self.no_of_doc_to_retrieve
            }).execute()
            # .execute send our request to Supabase

            return rpc_response

        except Exception as e:
            raise RuntimeError(f'data could not retrieved due to error {e}') from e


async def main():

    db = database_queries()
    await db.async_init()
        
        # Example: Store documents
    file_address = 'src/ingestion/LangGraph.pdf'

    #doc_objects = await PDF_parser(file_path=file_address) # Your document objects
    #await db.vector_embeddings_storage()
        
        # Example: Retrieve data
    results = await db.vector_data_retrieval("What are spellings of AI?")
    print(results)

if __name__ == "__main__":
   asyncio.run(main())


