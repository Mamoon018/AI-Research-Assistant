# Here we will be defining the schemas for every node. 

from pydantic import BaseModel, Field
from langchain.schema import Document


class BasestructureModel(BaseModel):
    class config:
        """ configuration of pydantic"""

        extra = "forbid"    # if any extra fields created by the LLM other than those defined in schema, this will raise the error.

# Node - 1 Schema
class PDF_Parser_schema(BasestructureModel):
    pdf_parser_results: list[Document]

# Node - 2 Schema
class Data_storage_schema(BasestructureModel):
    data_storage_status: str

