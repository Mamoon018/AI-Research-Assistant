
# Lets create a state class for the graph

from typing import Any, Literal
from pathlib import Path
from langgraph.graph import MessagesState
from langchain.schema import Document  # To store document in state object
from langchain_community.vectorstores import SupabaseVectorStore

class Keywordstate(MessagesState):

    # Input of the node-1: Question that user ask related to the document that will be uploaded.
    user_query: str 

    # Input of the node-1: Address of the document that user will specify to access the document.
    user_doc: Path 

    # Output of node-1: It contains the list of page-wise document objects containing page-content & metadata. For example: Documents: [Document(page_content="Text from page 1", metadata={"page": 1})]
    document_objects: list[Document]

    # Output of node-2: It contains the status of the vector embeddings storage. Goal is to keep record if vectors were stored successfully or not.
    Vector_storage_status: str

    # Output of node-3: It is the output of the router node. It will decide which node needs to be executed from here.
    router_decision: Literal["DB and LLM", "Web and LLM"]

    # Output of node - (4-A) : Retrieved_data contains the list of chunks with page-content & metadata.
    Retrieved_data: list[Document]

    # Output of node - (4-A): It stores the retrieved data and LLM analysis on the retrieved data.
    LLM_analysis_on_retrieved_data: str

    # Output of node - (4-B): It stores the LLM web-tool results plus LLM analysis on those results - in case user query is not related to the document uploaded.
    LLM_analysis_on_websearch_data: str

