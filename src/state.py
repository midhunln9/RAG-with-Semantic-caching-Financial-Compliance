from typing import TypedDict

class AgentState(TypedDict):
  query : str
  rewritten_query : str