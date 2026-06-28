from config import PROVIDER

from llm.ollama_provider import OllamaProvider


def get_provider():

    if PROVIDER == "ollama":
        return OllamaProvider()

    raise ValueError(
        f"Provider '{PROVIDER}' não suportado."
    )
