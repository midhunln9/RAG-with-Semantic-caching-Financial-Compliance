from abc import ABC, abstractmethod
from pydantic import BaseModel

class LLMStrategy(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, response_class : type[BaseModel] | None = None ) -> str | BaseModel:
        ...