from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import json


class CustomGLM(LLM):

    url: str
    temperature: float = 0.1
    max_length: int = 4096
    top_p: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "ChatGLM-6B-client"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        data = {"prompt": prompt,
                "temperature": self.temperature,
                "max_length": self.max_length,
                "top_p": self.top_p}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        return response['response']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"url": self.url}


if __name__ == "__main__":
    llm = CustomGLM(url="http://127.0.0.1:8000/")
    print(llm("刑法第一条是什么？"))
