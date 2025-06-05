
# Import BaseSetting, Secretstr, finddotenv, os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from dotenv import find_dotenv
import os 

class settings(BaseSettings):

    """
    This class inherits the Basesettings to facilitate defining "settingconfiDict" that allows us to decide
    from where, how, and what to look for the variables. 

    We aim to define environment variables as attribute of setting - by using SecretStr to store values in it. All
    values of attributes are stored securely using SecretStr.    

    **Attributes:**

        **LLMs APIs:**
        GEMINI_API_KEY (SecretStr | None = None): API key for Gemini LLM services
        OPENAI_API_KEY (SecretStr | None = None): API key for OpenAI LLM services
        GROQ_API_KEY (SecretStr | None = None): API key for GROQ LLM services
        MISTRAL_API_KEY (SecretStr | None = None): API key for Mistral AI LLM services

        **Web Search Tools APIs:**
        EXA_API_KEY (SecretStr | None = None): API key for Exa web search services
        TAVILY_API_KEY (SecretStr | None = None): API key for Tavily web search services

        **Monitoring/Observability Tools:**
        OPIK_WORKSPACE (SecretStr | None = None): Name of Opik workspace
        OPIK_PROJECT_NAME (SecretStr | None = None): Name of the Opik Project
        OPIK_API_KEY (SecretStr | None = None): API key for the Opik observability services
    
    """
    # We will define app_config to read .env according to given parameters.
    model_config = SettingsConfigDict(
        env_file= find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="allow",
    )

    # LLMs
    GEMINI_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    MISTRAL_API_KEY: SecretStr | None = None

    # Web search tool
    EXA_API_KEY: SecretStr | None = None
    TAVILY_API_KEY: SecretStr | None = None

    # Monitoring / Observability tool
    OPIK_WORKSPACE: SecretStr | None = None
    OPIK_PROJECT_NAME: SecretStr | None = None
    OPIK_API_KEY: SecretStr | None = None 


# instantiate the class -- we can use this instance to get any API.
settings = settings()
 
# ****************************************************************
# Defining Function that will fetch required API
# ****************************************************************

def get_api_key(api_key: SecretStr | str | None) -> str | None:
    """
    Function fetches the API key value from the class attributes.

    Function will take api_key as input - Example: settings.GEMINI_API_KEY, and uses "get_secretvalue()" to
    fetch secret value of API key.

    Args:
    api_key (SecretStr | None): Secret key Object

    Returns:
    (str | None): API value of Secret object
    
    Raises:
    "If API key is not set or any error in retrieving it"
    
    """
    try:
        if api_key is None:
            return None
        if isinstance(api_key,SecretStr):
            return api_key.get_secret_value()
        return api_key
    except Exception as e:
        raise ValueError(f"API Key not found ({api_key}): {e} ") from e
    
