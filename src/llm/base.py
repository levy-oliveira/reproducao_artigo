from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generates a prediction."""
        pass
