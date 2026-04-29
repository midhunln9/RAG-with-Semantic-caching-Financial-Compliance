from src.llm.nvidia import NVIDIALLM
from src.llm.openai import OpenAILLM

REGISTERED_LLM_STRATEGIES = {
    "openai": OpenAILLM,
    "nvidia": NVIDIALLM,
}


def get_llm_strategy(strategy_name: str):
    strategy_class = REGISTERED_LLM_STRATEGIES.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"LLM strategy '{strategy_name}' is not registered.")
    return strategy_class()
