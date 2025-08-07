import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pyagenity.agent.agent import Agent


def main():
    agent = Agent(
        name="GeminiDemo",
        model="gemini-2.0-flash",
        custom_llm_provider="gemini",
    )
    prompt = "How are you?"
    response = agent.run(prompt)
    print("Content:", response.content)
    print("Thinking:", response.thinking)
    print("Usage:", response.usage)


if __name__ == "__main__":
    main()
