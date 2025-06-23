
"""
We are going to define all nodes for the agent in this nodes.py.
"""

import json 
from typing import Any 

from src.agents.research_assistant.state import Keywordstate   # Import State
from langchain_core.messages import HumanMessage # To pass the prompt to LLM
from langgraph.config import get_stream_writer # To show custom message in streaming 



