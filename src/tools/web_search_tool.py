# Here we will define web search tool and will parse its output as well.

from src.utils.model_initializer import tavily_client_setup, Exa_client_setup
from langchain_core.tools import tool , BaseTool
from langchain.callbacks import StdOutCallbackHandler
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional

"""
1- Firstly, we will define Class BaseModel to define data types of input parameters.
2- Within the class, using tool decorator we will define _run & _arun functions incorporating both Tavily & Exa. 
"""

class input_params_websearch(BaseModel):
    """
    We will define the input parameters for Tavily & Exa - along with their datatypes so, that we can ensure that 
    correct parameters get stored in input parameters.
    
    """
    query: str = Field(
        description= 'Query to look up for'
    )

    max_results: int = Field (
        description= 'Maximum number of search results that will be returned'
    )

    search_depth: Optional[Literal['basic', 'advance']] = Field(
        description= 'It refers to the type of search we want to carry out - basic level or advance level'
    )

    time_range: Optional[Literal['day', 'week', 'month', 'year']] = Field(
        description= 'It defines the time range of search execution'
    )

    include_answer: bool = Field(
        description= 'Along with the search, it provides the short answer to it. By default it is set to False'
    )

class web_search(BaseTool):
    name: str = 'web search tool'
    description: str = 'Web searcht tool for performing search using Tavily with fallback to Exa on error'
    args_schema: str = input_params_websearch

    def params_tavily_Exa(self):
        """
        It defines the paramters for Tavily & Exa. We can extract and pass the relevant parameters from the 
        list and pass it to the respectiv search tool.
        """

        params = {
                'query': 'When did Pakistan come into existence?',
                'max_results': 3,
                'search_depth': 'basic',
                'time_range': 'year',
                'include_answer': True,
                  }   # need to include parameters for exa as well. This will become list of two dictionaries
        return params
    
    def _run(self):
        
        parameters = self.params_tavily_Exa()
        parsed_input = input_params_websearch(**parameters)
        Tavily_client = tavily_client_setup()
        Exa_client = Exa_client_setup()
        
        try:
            results = Tavily_client.search(**parsed_input.model_dump())
            return results
        except Exception as e:
            try:
                results = Exa_client.search_and_contents(**parsed_input.model_dump()) # need to pass exa parameters separately as they are different from tavily

                return results
            except Exception as e:

                return f'error occured {e}'
            
#if __name__ == "__main__":
#    tool = web_search()
#    response = tool._run()  
#    print(response)


