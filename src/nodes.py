from src.schemas.rewrite_query_json_call import RewriteQueryJsonCall
from src.services import RagWorkflowService
from src.state import AgentState
from src.prompts.rewrite_prompt import QUERY_REWRITE_PROMPT

class Nodes:
  def __init__(self, service : RagWorkflowService):
    self.service = service
  
  async def rewrite_query(self, state : AgentState) -> dict:
    query = state["query"]
    prompt = QUERY_REWRITE_PROMPT.format(user_query=query)
    rewritten_query = await self.service.rewrite_query_service(prompt = prompt, response_class=RewriteQueryJsonCall)
    return {"rewritten_query": rewritten_query}
  