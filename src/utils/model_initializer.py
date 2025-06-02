# In this file, we will initialize the LLMs with their parameters and Tools SDK that will be used throughout the app.

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.settings import settings,get_api_key


def get_openai_llm(model_num: int = 1, temperature: int = 0.5):
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
    model_num = {
        1: "gpt-4.1-mini",
        2: "o4-mini-2025-04-16" }
    
    # Let's get openai api key for llm
    openai_api_key = get_api_key(settings.OPENAI_API_KEY)
    
    if model_num == 1:
        openai_llm = ChatOpenAI(
            model= "gpt-4.1-mini",
            temperature= temperature,
            max_completion_tokens= 400,
            timeout= None,
            max_retries= 2,
            api_key= openai_api_key
        )
    else:
        openai_llm = ChatOpenAI(
            model= "o4-mini-2025-04-16",
            temperature= temperature,
            max_completion_tokens= 400,
            timeout= None,
            max_retries= 2,
            api_key= openai_api_key
        )
    return openai_llm


def get_gemini_llm(model_num: int = 1, temperature: int = 0.5):
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
    gemini_api_key = get_api_key(settings.GEMINI_API_KEY)

    gemini_models_list = {1:"gemini-2.0-flash", 2: "gemini-2.5-flash-preview-04-17",
                     3: "gemini-2.5-pro-exp-03-25", 4: "gemini-1.5-pro"}
    gemini_model = gemini_models_list[model_num]

    if model_num == 1:
        gemini_llm = ChatGoogleGenerativeAI(
            model = gemini_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key = gemini_api_key
        )
    return gemini_llm
    




    
