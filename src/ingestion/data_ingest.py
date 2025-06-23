# In this file we will define the function that will take the PDF and parse it and finally store it into the
# Supabase vector database. 
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from typing import Any 
from pathlib import Path 


async def PDF_parser(file_path: Path) -> list[Any]:
    
    """
    Takes the file path and parse it using PDFloader, then store content of all pages in the list 'pages'.

    **Args:**
    file_path (Any): File address is the input for the function, that will be used to access the file.

    **Returns:**
    pages (list[Any]): It is the list that contains the document objects for all pages of the PDF. Document object
    contains page_content & meta_data.

    """

    pages: list[Any] = []
    loader = PyPDFLoader(file_path= file_path)
    try:
        async for page in loader.alazy_load():
            pages.append(page)
        return pages
    except Exception as e:
        raise RuntimeError(f'PDF could not parsed as error occurred {e}') from e


            #document_page_content = [ {'meta_data' : doc.metadata} for doc in pages]
    #return document_page_content

#if __name__ == '__main__':
#    file_address = 'src/ingestion/LangGraph.pdf'
#    results = asyncio.run(PDF_parser(file_path=file_address))
#    print(results)

