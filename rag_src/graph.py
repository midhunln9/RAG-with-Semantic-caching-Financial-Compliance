from langgraph.graph import END, START, StateGraph

from rag_src.nodes import Nodes
from rag_src.state import AgentState


class Graph:
    def __init__(self, nodes: Nodes):
        self.nodes = nodes
        self.graph = StateGraph(AgentState)

    def build_graph(self):
        self.graph.add_node("guard_prompt", self.nodes.guard_prompt)
        self.graph.add_node("return_off_topic_response", self.nodes.return_off_topic_response)

        # After the topicality guard passes, rewrite so the cache key is
        # based on the normalized query.
        self.graph.add_node("rewrite_query", self.nodes.rewrite_query)
        self.graph.add_node("check_cache", self.nodes.check_cache)
        self.graph.add_node("return_cached_answer", self.nodes.return_cached_answer)
        self.graph.add_node("cache_miss", self.nodes.cache_miss)

        # Miss path: run the existing RAG workflow.
        self.graph.add_node("retrieve_docs", self.nodes.retrieve_documents)
        self.graph.add_node("get_conversations", self.nodes.get_conversations)
        self.graph.add_node("rag_answer", self.nodes.rag_answer)
        self.graph.add_node("store_answer_in_cache", self.nodes.store_answer_in_cache)

        self.graph.add_edge(START, "guard_prompt")
        self.graph.add_conditional_edges(
            "guard_prompt",
            self._route_after_topic_guard,
            {
                "on_topic": "rewrite_query",
                "off_topic": "return_off_topic_response",
            },
        )

        self.graph.add_edge("return_off_topic_response", END)
        self.graph.add_edge("rewrite_query", "check_cache")
        self.graph.add_conditional_edges(
            "check_cache",
            self._route_after_cache_lookup,
            {
                "cache_hit": "return_cached_answer",
                "cache_miss": "cache_miss",
            },
        )

        self.graph.add_edge("return_cached_answer", END)
        self.graph.add_edge("cache_miss", "retrieve_docs")
        self.graph.add_edge("cache_miss", "get_conversations")
        self.graph.add_edge(["retrieve_docs", "get_conversations"], "rag_answer")
        self.graph.add_edge("rag_answer", "store_answer_in_cache")
        self.graph.add_edge("store_answer_in_cache", END)

        return self.graph.compile()

    @staticmethod
    def _route_after_cache_lookup(state: AgentState) -> str:
        return "cache_hit" if state.get("cache_hit", False) else "cache_miss"

    @staticmethod
    def _route_after_topic_guard(state: AgentState) -> str:
        return "on_topic" if state.get("is_on_topic", False) else "off_topic"
