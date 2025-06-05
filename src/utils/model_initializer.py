# In this file, we will initialize the LLMs with their parameters and Tools SDK that will be used throughout the app.

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from typing import Literal, Callable, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient , AsyncTavilyClient
from exa_py import Exa , AsyncExa
from src.utils.settings import settings,get_api_key


# Openai LLM
def get_openai_llm(model_num: Literal[1,2] = 1 , temperature: int = 0.5):
    """
    It contains multiple openai models under usage and can switch between 
    by specifying just a model number. It will be useful when we would have to invoke two different openai models
    in different nodes.

    **Args:**
    model_num (int): Model number to choose the model that we want to use. Default value is 1 which means our default 
    openai model will be "gpt-4.1-mini".
    Other options in openai models: 
    --> "o4-mini-2025-04-16" 

    **Returns:**
    llm (ChatOpenAI): Openai llm initialized instance 
    """
    model_list = {
        1: "gpt-4.1-mini",
        2: "o4-mini-2025-04-16" }
    
    # Let's get openai api key for llm
    openai_api_key: str | None = get_api_key(settings.OPENAI_API_KEY)
    if openai_api_key is None:
        return "Openai api key is not found"
    
    openai_model = model_list[model_num]

    openai_llm = ChatOpenAI(
            model= openai_model,
            temperature= temperature,
            max_completion_tokens= 400,
            timeout= None,
            max_retries= 2,
            api_key= openai_api_key
        )
    return openai_llm


# Gemini LLM
def get_gemini_llm(model_num: Literal[1,2,3,4] = 1, temperature: int = 0.5):
    """
    It contains multiple gemini models that can be used in our project. To choose between different models 
    we just need to specify the model_num (model number).

    **Args:**
    model_num (int): Model number represent that model of gemini we want to use. we can switch between following:
    (1) "gemini-2.0-flash"
    (2) "gemini-2.5-flash-preview-04-17"
    (3) "gemini-2.5-pro-exp-03-25"
    (4) "gemini-1.5-pro"

    **Returns**
    ChatGoogleGenerativeAI: Initialized instance of gemini llm
    """
    
    # lets get gemini api key 
    gemini_api_key: str | None = get_api_key(settings.GEMINI_API_KEY)
    if gemini_api_key is None:
        return "Gemini api key is not found"


    gemini_models_list = {
        1:"gemini-2.0-flash", 
        2: "gemini-2.5-flash-preview-04-17",
        3: "gemini-2.5-pro-exp-03-25", 
        4: "gemini-1.5-pro"
                            
                            }
    
    gemini_model = gemini_models_list[model_num]

    gemini_llm = ChatGoogleGenerativeAI(
            model = gemini_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key = gemini_api_key
        )
    return gemini_llm


# Groq LLM
def get_groq_llm(model_num: Literal[1,2,3], temperature: int = 0.5) -> ChatGroq:
    """
    This function initializes the Groq llm that will be called in the nodes. 

    **Args:**
    model_num (int): It takes the integar as input that represent the model number associated with list of models 
    in the dictionary. We can change it to switch among the following models.
    (1) "llama3-70b-8192"
    (2) "deepseek-r1-distill-llama-70b"
    (3) "meta-llama/llama-4-scout-17b-16e-instruct"

    temperature (int): It is the input to control the randomness of the model. Default value is set to 0.5

    **Returns:**
    ChatGroq: Initialized object of Groq llm.
    """

    # Lets get the Groq API
    GROQ_API: str | None = get_api_key(settings.GROQ_API_KEY)
    if GROQ_API is None:
        return "GROQ API is not found"

    # Lets define the dictionary that contains the Groq models we can use.
    Groq_model_list = {
        1: "llama3-70b-8192",
        2: "deepseek-r1-distill-llama-70b",
        3: "meta-llama/llama-4-scout-17b-16e-instruct"
                }
    groq_model = Groq_model_list[model_num]

    groq_llm = ChatGroq(
        model= groq_model,
        temperature= temperature,
        max_tokens= None,
        timeout= None,
        max_retries= 2,
        api_key= GROQ_API
    )
    return groq_llm


# Tavily web search client
def tavily_client_setup(asyncronous_tavily: bool = False) -> TavilyClient | AsyncTavilyClient:
    """
    Function returns the Tavily client. It can return both Syncronous and Asyncronous client

    **Args:**
    asyncronous_tavily (bool): It is set to False by default means we will get syncronous client & if passes True 
                                it will return Asyncronous client.
    
    **Returns:**
    TavilyClient or AsyncTavilyClient
    """

    TAVILY_API: str | None = get_api_key(settings.TAVILY_API_KEY)
    if TAVILY_API is None:
        return 'Tavily API is not found'

    if asyncronous_tavily:
        return AsyncTavilyClient(api_key= TAVILY_API)
    else:
        return TavilyClient(api_key=TAVILY_API)
    

# Exa web search Client 
def Exa_client_setup(asyncronous_exa: bool = False) -> Exa | AsyncExa:
    """
    Function returns the Exa client. It can return both Syncronous and Asyncronous client

    **Args:**
    asyncronous_exa (bool): It is set to False by default means we will get syncronous client & if passes True 
                                it will return Asyncronous client.
    
    **Returns:**
    Exa or AsyncExa
    """
    
    Exa_API: str | None = get_api_key(settings.EXA_API_KEY)
    if Exa_API is None:
        return "Exa API is not found"

    if asyncronous_exa:
        return AsyncExa(Exa_API)
    else:
        return Exa(Exa_API)
    

# Function that will be used to invoke models with fallbacks

"""
    invoking single model:
    response: Structured_output = llm1.invoke(message)
    invoking model with fallback:
    response: Strcutured_output = llm1.with_fallbacks([llm2])

    llm1 & llm2 --> These are the chat-Model defined with parameters. 

    # What do we want to implement?
    Initializes a primary model with optional structured output and tool binding,
    and sets up fallback models with the same options.

    **llm with fallback that gives structured output - tools binded**
    Example to define llm binded with tools to give structured output:
    llm_t = llm.bind_tools([t1, t2])
    llm_t_structured = llm_t.with_structured_output(MySchema)  --> This structures the output of LLM, not the tool's output.

    What if we want to get output of tool in structured form?
    We need to define it in the tool-function so, that when tool gets executed function returns tool output in the 
    structured form. 

"""
