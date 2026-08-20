from typing import Optional

from core.agent import Agent
from core.llm import HelloAgentsLLM
from core.config import Config
from core.message import Message


class SimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, **kwargs) -> str:
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

        return response

    def stream_run(self, input_text: str, **kwargs):
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
