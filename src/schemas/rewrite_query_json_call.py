from pydantic import BaseModel, Field

class RewriteQueryJsonCall(BaseModel):
  rewritten_query : str = Field(description="A concise rewritten query for proper retrieval of documents in the field of financial compliance. Return only the rewriten query.")