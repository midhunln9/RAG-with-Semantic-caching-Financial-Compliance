from langgraph.graph import StateGraph, START, END
from src.nodes import Nodes
from src.state import AgentState


class Graph:
  def __init__(self, nodes : Nodes):
    self.nodes = nodes
    self.graph = StateGraph(AgentState)
  
  def build_graph(self) -> StateGraph:
    self.graph.add_node("rewrite_query", self.nodes.rewrite_query)
    self.graph.add_edge(START, "rewrite_query")
    self.graph.add_edge("rewrite_query", END)
    return self.graph.compile()
