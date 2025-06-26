
import asyncio
from langgraph.graph import StateGraph
from IPython.display import Image, display

from src.agents.research_assistant.state import Keywordstate
from pathlib import Path
from langchain.schema import Document
from src.ingestion.data_ingest import PDF_parser
from src.tools.supabase_tool import database_queries

from src.agents.research_assistant.schemas import PDF_Parser_schema, Data_storage_schema


# Lets create the first node for the graph - PDF Parser Node
async def user_pdf_parsed(state:Keywordstate):
    """
    Receives the input document & query from the user, and parse the document using pdf.alazy()

    This function receives the pdf document & user question related to the document as input. It stores the 
    question to the state, and parse the user document using pdf parser. It will generate the document objects
    that will be used for vector embeddings which will be further used to store in database.


    **Returns:**
    document_objects : (list[Document]) list of the document objects
    """

    # lets get the user document from the state
    user_uploaded_document: Path = state["user_doc"]

    # lets intialize the list to store parsed document objects
    parsed_pdf_doc_objects: list[Document] = []

    try:
        # lets intialize the tool 
        pdf_parser_results: PDF_Parser_schema = await PDF_parser(user_uploaded_document)

        # lets extract the variable from the object created based on the schema 
        parsed_pdf_doc_objects = pdf_parser_results

        # lets update the state and print the results

        return {"document_objects": parsed_pdf_doc_objects}

    except Exception as e:
        raise print(f'Error occured {e}') from e


# Lets create the 2nd node for the graph - Data Storage Node
async def vector_storage_supabaseDB(state:Keywordstate):

    """
    This functions takes the parsed document objects and create embeddings of them, then store those embeddings
    in Supabase Vector database.

    **Args:**
    parsed_doc_objects (list[Document]): It is the list of the document objects that wil be to create vector embeddings.
    
    **Returns:**
    It just stores tyhe data into vector database and raises the status of successful storage of data.

    **Raises:**
    It raises the error if either embeddings are not created or data is not being able to store in the Supabase vector database.
    """

    # Lets get the document objects from the state
    doc_objects: list[Document] = state["document_objects"]

    # lets define the output variable 
    Vector_storage_status: str = None

    try:
        # lets get the tool function from the supabasetool file 
        DB_methods = database_queries()
        await DB_methods.async_init()

        # lets define the behavior of the node 
        data_storage: Data_storage_schema = await DB_methods.vector_embeddings_storage(doc_objects)

        # lets extract the output from the object creared based on the schema
        Vector_storage_status = data_storage.data_storage_status

        # lets update the state
        return {"Vector_storage_status": Vector_storage_status}, print("data stored successfully!")
    
    except Exception as e:
        raise e






# Lets test these two nodes first then we will proceed towards the router node
# Define the state of graph
builder = StateGraph(Keywordstate)

# Add nodes
builder.add_node("user_pdf_parsed", user_pdf_parsed)
builder.add_node("vector_storage_supabaseDB", vector_storage_supabaseDB)

# Add edges to connect the nodes
builder.add_edge("user_pdf_parsed", "vector_storage_supabaseDB")
builder.set_entry_point("user_pdf_parsed")

# compile the graph
workflow = builder.compile()

# User input & invoke the graph
user_input_for_entry_node = {"user_query": "what are AI Agents?", "user_doc": "C:\\AI Research Assistant Code\\src\\ingestion\\LangGraph.pdf"}
asyncio.run(workflow.ainvoke(user_input_for_entry_node))

# lets get the picture of the graph
#display(Image(workflow.get_graph().draw_mermaid_png()))