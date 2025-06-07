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


async def vector_embeddings(doc_objects):
    # Lets create chunks of the list of objects created by PDFParser
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
    chunks = text_splitter.split_documents(doc_objects)

    openai_embeddings = OpenAIEmbeddings()
    supabase_client_activated = supabase_client()

    try: 
    # Let's store the chunks in DB
        vector_store = SupabaseVectorStore.from_documents(
            documents= chunks,
            embedding= openai_embeddings,
            client = supabase_client_activated
        )
    except Exception as e:
         raise RuntimeError(f'Vectorembeddings could not store because of error {e}') from e   
    
    # we can use the as_retriever() to extract from DB.
    print('embedding stored')
    

async def main():
    file_address = 'src/ingestion/LangGraph.pdf'
    list_doc_objs = await PDF_parser(file_path=file_address)

    await vector_embeddings(doc_objects=list_doc_objs)

if __name__ == '__main__':
    asyncio.run(main()) 