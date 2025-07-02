
import asyncio
from langgraph.graph import StateGraph
from IPython.display import Image, display
import opik
opik.configure(use_local=False)
from opik.integrations.langchain import OpikTracer
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from typing import Any

from src.agents.research_assistant.state import Keywordstate
from pathlib import Path
from langchain.schema import Document
from src.ingestion.data_ingest import PDF_parser
from src.tools.supabase_tool import database_queries

from src.agents.research_assistant.schemas import PDF_Parser_schema, Data_storage_schema, Router_node_schema
from src.agents.research_assistant.prompts import ROUTER_NODE_PROMPT
from src.utils.model_initializer import INITIALIZING_MODELS_STRUCTUREDOUTPUT_TOOLS
from src.utils.model_initializer import get_openai_llm, get_gemini_llm, get_groq_llm

### Lets define the Models for nodes here ###

MODEL_WITH_FALLBACK_ROUTER = INITIALIZING_MODELS_STRUCTUREDOUTPUT_TOOLS(
    primary_model_fn= get_openai_llm,
    primary_model_fn_kwargs= {"model_num": 1, "temperature" :0.5},
    fallback_model_fn= get_openai_llm,
    fallback_fn_kwargs= [{"model_num" :1}, {"temperature" :0.5}],
    bind_tool_list= [],
    structured_output= Router_node_schema
)







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

    # lets initialize the list for storing page content of each doc object
    extracted_page_contents: list[Any] = []

    try:
        # lets intialize the tool 
        pdf_parser_results: PDF_Parser_schema = await PDF_parser(user_uploaded_document)

        # lets extract the variable from the object created based on the schema 
        parsed_pdf_doc_objects = pdf_parser_results

        # Lets extract the page_content from the doc_objects stored in list
        for doc in parsed_pdf_doc_objects:
            extracted_page_contents.append(doc.page_content)

        # lets update the state and print the results

        return {"document_objects": parsed_pdf_doc_objects,
                "extracted_page_contents": extracted_page_contents}

    except Exception as e:
        raise RuntimeError(f'Error occured {e}') from e


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
        await DB_methods.async_init()  # Initializing the attributes of class.

        # lets define the behavior of the node 
        data_storage = await DB_methods.vector_embeddings_storage(doc_objects)

        # lets extract the output from the object creared based on the schema
        if data_storage:

            Vector_storage_status = 'Vectors stored successfully'

        print(Vector_storage_status)

        # lets update the state
        return {"Vector_storage_status": Vector_storage_status}
    
    except Exception as e:
        raise e

async def router_node(state:Keywordstate):

    """
    It takes the doc_objects as an input and extract the content of all objects. Then that object passes to the LLM
    that ckecks if information exist in the content or not based on the user query.

    **Args:**
    user_query (str): It is the user query stored in the state. It is the question that needs to be answers.
    user_doc (list[Document]): It is the document objects containing meta_data and page_content

    **Returns:**
    router_decision (Literal["DB and LLM", "Web and LLM"]): It returns the name of the type of node (After router we have 3 different nodes) that router needs to initiate as a result of decision

    **Raises:**
    It raises error if LLM does not get initiated!
    
    """

    """
    1) Input data
    2) Prompt of LLM
    3) Schema of LLM
    4) Initialize the OUTPUT object 
    5) initialize the model with fallacks binded by tools while assigning schema
    6) fetch the data from the schema object and store in the output object initialized earlier
    7) make changes if required. 
    8) update the state
    
    """

    # lets get the input objects from the state
    document_objects: list[Document] = state["document_objects"]
    user_query: str = state["user_query"]
    page_contents: list[Any] = state["extracted_page_contents"]

    # lets get the prompt for the node
    router_prompt = PromptTemplate(
        input_variables= ["user_question", "content_of_doc_objects"] ,
        template= ROUTER_NODE_PROMPT
    )

    final_router_prompt = router_prompt.format(user_question = user_query, content_of_doc_objects = page_contents)


    # lets initialize the object to store the decision of router
    router_decision: str = None

    # lets initialize the object to store the reasoning of llm behind choosing category for user query
    router_reasoning: str 

    try:
        router_output: Router_node_schema = await MODEL_WITH_FALLBACK_ROUTER.ainvoke([HumanMessage(content= final_router_prompt)])

        # lets extract the output of the llm
        router_decision: str = router_output.router_call
        router_reasoning: str = router_output.router_reasoning_data

        # lets update the state in return part
        return {
            "router_decision": router_decision,
            "router_reasoning": router_reasoning
        }
    except Exception as e:
        raise RuntimeError(f'error occurred due {e}') from e 






# Lets test these two nodes first then we will proceed towards the router node
# Define the state of graph
builder = StateGraph(Keywordstate)

# Add nodes
builder.add_node("user_pdf_parsed", user_pdf_parsed)
builder.add_node("vector_storage_supabaseDB", vector_storage_supabaseDB)
builder.add_node("router_decision_node", router_node)

# Add edges to connect the nodes
builder.add_edge("user_pdf_parsed", "vector_storage_supabaseDB")
builder.add_edge("vector_storage_supabaseDB", "router_decision_node")
builder.set_entry_point("user_pdf_parsed")

# compile the graph
workflow = builder.compile()

# User input & invoke the graph
#user_input_for_entry_node = {"user_query": "what are AI Agents?", "user_doc": "C:\\AI Research Assistant Code\\src\\ingestion\\LangGraph.pdf"}
#asyncio.run(workflow.ainvoke(user_input_for_entry_node))
#print()

# Lets call the graph with Opik 
tracer = OpikTracer(graph=workflow.get_graph(xray=True))
inputs = {"user_query": "what are AI Agents?", "user_doc": "C:\\AI Research Assistant Code\\src\\ingestion\\LangGraph.pdf"}
result = asyncio.run(workflow.ainvoke(inputs,config={"callbacks": [tracer]}))
print(result["router_reasoning"])


# lets get the picture of the graph
#display(Image(workflow.get_graph().draw_mermaid_png()))