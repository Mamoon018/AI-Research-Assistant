
# Lets create a state class for the graph

from typing import Any, Literal
from langgraph.graph import MessagesState
from langchain.schema import Document  # To store document in state object
from langchain_community.vectorstores import SupabaseVectorStore

class Keywordstate(MessagesState):

    # Input for node-1: User question related to the document that will be uploaded.
    user_query: str 

    # Input for node-1: Document that user will be uploading & want to inquire about.
    user_doc: list[Document]

    # Output of node-1: It contains the list of page-wise document objects containing page-content & metadata. For example: Documents: [Document(page_content="Text from page 1", metadata={"page": 1})]
    Documents: list[Document]

    # Output of node-1: It contains the list of chunk-wise document objects containing page-content & metadata.
    Chunks: list[Document]

    # Output of node-2: vectorstore contains the details of Supabase table & data in it.
    Vectorstore: SupabaseVectorStore

    # Output of node-3: It labels the query as if it requires DB,LLM or both.
    Router_decision: Literal["DB", "LLM", "DB and LLM"]

    # Output of node-4/5: Retrieved_data contains the list of chunks with page-content & metadata.
    Retrieved_data: list[Document]

    # Output of node-5: It stores the retrieved data and LLM analysis on the retrieved data.
    LLM_analysis_on_retrieved_data: str 

    # Output of node-6: It stores the LLM web-tool results plus LLM analysis on those results - in case user query is not related to the document uploaded.
    LLM_generated_analysis: str 

