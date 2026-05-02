from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMStrategy(ABC):
    @property
    @abstractmethod
    def cache_llm_string(self) -> str: ...

    @abstractmethod
    def generate_response(
        self, prompt: str, response_class: type[BaseModel] | None = None
    ) -> str | BaseModel: ...
