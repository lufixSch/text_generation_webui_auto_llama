from abc import ABC, abstractmethod

import requests as req


class LLMInterface(ABC):
    """Generic interface to communicate with a LLM"""

    @abstractmethod
    def completion(self, prompt: str) -> str:
        raise NotImplementedError(
            "The `completion` method needs to be implemented by each LLM Interface"
        )


class OobaboogaLLM(LLMInterface):
    """LLM Interface calling the oobabooga api for text generation"""

    def __init__(
        self,
        api_endpoint: str,
    ):
        self.api_endpoint = api_endpoint

    def completion(
        self,
        prompt: str,
        stopping_strings: list[str] = [],
        temperature: float = 0.5,
        max_new_tokens: int = 200,
    ) -> str:
        url = f"{self.api_endpoint}/api/v1/generate"
        body = {
            "prompt": prompt,
            "stopping_strings": stopping_strings,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

        res = req.post(url, json=body)

        if res.status_code != 200:
            raise ValueError(f"LLM Completion failed with code {res.status_code}")

        res_dict = res.json()
        text = res_dict["results"][0]["text"]

        return text
