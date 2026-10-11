import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage


class MainAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],operator.add]
    user_info: dict
    next_agent:str
    active_agent:str
    handoff_count:int