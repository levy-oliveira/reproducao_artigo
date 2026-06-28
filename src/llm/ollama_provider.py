import requests

from config import MODEL_NAME
from llm.base import LLMProvider
from prompts import SYSTEM_PROMPT


class OllamaProvider(LLMProvider):

    URL = "http://localhost:11434/api/chat"

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(
            self.URL,
            json=payload,
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        # print("\n===== OLLAMA RESPONSE =====")
        # print(data)
        # print("===========================\n")

        # return _clean_response(
        #     data["message"]["content"]
        # )
        return {
            "prediction": data["message"]["content"].strip(),
            "model": data["model"],
            "prompt_tokens": data["prompt_eval_count"],
            "completion_tokens": data["eval_count"],
            "latency_ms": data["total_duration"] / 1_000_000,
        }


def _clean_response(text: str) -> str:
    text = text.strip()
    text = text.replace("```", "")

    return text
