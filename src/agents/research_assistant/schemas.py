# Here we will be defining the schemas for every node. 

from pydantic import BaseModel, Field
from langchain.schema import Document
from typing import Literal, Any


class BasestructureModel(BaseModel):
    class config:
        """ configuration of pydantic"""

        extra = "forbid"    # if any extra fields created by the LLM other than those defined in schema, this will raise the error.

# Node - 1 Schema
class PDF_Parser_schema(BasestructureModel):
    pdf_parser_results: list[Document] = Field(
        ..., description= "It stores the document objects of the file uploaded"
    )

# Node - 2 Schema
class Data_storage_schema(BasestructureModel):
    data_storage_status: str = Field(
        ..., description= "It is the status that will confirm if data storage in the database has been executed successfully or not",
    )

# Node - 3 Schema
class Router_node_schema(BasestructureModel):
    router_call: Literal["DB and LLM", "Web and LLM"] = Field(
        ..., description= "It can only give two possible output based on which LLM will decide which node needs to be executed moving forward" ,
    )

    router_reasoning_data: str  = Field(
        ..., description = "It stores the reason of the decision why LLM choose one of two options for categorization of user query"
    )
