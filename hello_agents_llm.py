import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class HelloAgentsLLM:
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError(
                "Model ID, API key, and base URL must be provided or defined in the .env file.")

        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        print(f"Thinking by {self.model} ...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            return "".join(collected_content)

        except Exception as e:
            print(f"Error while calling LLM API: {e}")
            return None


if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()

        example_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code."
            },
            {
                "role": "user",
                "content": "写一个快速排序算法"
            }
        ]

        response_text = llm_client.think(example_messages)
        if response_text:
            print("\n\n Model Response:")
            print(response_text)

    except ValueError as e:
        print(e)
