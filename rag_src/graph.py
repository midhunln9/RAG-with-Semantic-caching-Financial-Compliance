from langgraph.graph import END, START, StateGraph

from rag_src.nodes import Nodes
from rag_src.state import AgentState


class Graph:
    def __init__(self, nodes: Nodes):
        self.nodes = nodes
        self.graph = StateGraph(AgentState)

    def build_graph(self):
        # Branch A: query rewrite -> document retrieval
        self.graph.add_node("rewrite_query", self.nodes.rewrite_query)
        self.graph.add_node("retrieve_docs", self.nodes.retrieve_documents)

        # Branch B: load past conversation history
        self.graph.add_node("get_conversations", self.nodes.get_conversations)

        # Join: final RAG answer using both branches' outputs
        self.graph.add_node("rag_answer", self.nodes.rag_answer)

        # Both branches start in parallel from START
        self.graph.add_edge(START, "rewrite_query")
        self.graph.add_edge(START, "get_conversations")

        # Branch A is sequential within itself
        self.graph.add_edge("rewrite_query", "retrieve_docs")

        # Join the two branches so rag_answer runs only after both complete.
        self.graph.add_edge(["retrieve_docs", "get_conversations"], "rag_answer")

        self.graph.add_edge("rag_answer", END)

        return self.graph.compile()
