from typing import Any

from llm_sdk import Small_LLM_Model


class Answer:
    def __init__(self) -> None:
        self.model = Small_LLM_Model(device="cpu")

    def run(
        self,
        prompts: list[dict[str, Any]],
        definitions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in prompts:
            results.append(
                {"prompt": item["prompt"], "name": "", "parameters": {}}
            )
        return results
