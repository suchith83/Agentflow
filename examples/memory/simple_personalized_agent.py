"""
Simple Personalized Agent Example using TAF + Mem0 + Cloud Qdrant

A streamlined example showing basic integration between:
- TAF for agent framework
- Mem0 for memory management
- Cloud Qdrant for vector storage
- LiteLLM for model calls

This example demonstrates a chatbot that remembers user preferences and conversation history.
"""

import asyncio
import os

from dotenv import load_dotenv
from litellm import acompletion
from mem0 import Memory

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph import StateGraph
from agentflow.state import AgentState, Message
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages


# Load environment variables
load_dotenv()

# Set environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["MEM0_API_KEY"] = os.getenv("MEM0_API_KEY", "")


class MemoryAgentState(AgentState):
    """State with user ID for memory context."""

    user_id: str = ""


class SimplePersonalizedAgent:
    """Simple personalized agent using Mem0 for memory."""

    def __init__(self):
        # Mem0 configuration for cloud Qdrant (768 dimensions for Gemini)
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "simple_agent_memory",
                    "url": os.getenv("QDRANT_URL"),
                    "api_key": os.getenv("QDRANT_API_KEY"),
                    "embedding_model_dims": 768,  # For Gemini embeddings (768 dimensions)
                },
            },
            "llm": {
                "provider": "gemini",
                "config": {"model": "gemini-2.0-flash-exp", "temperature": 0.1},
            },
            "embedder": {"provider": "gemini", "config": {"model": "models/text-embedding-004"}},
        }

        self.memory = Memory.from_config(config)
        self.app_id = "simple-agent"
        self._build_graph()

    def _build_graph(self):
        """Build TAF graph."""
        graph = StateGraph[MemoryAgentState](MemoryAgentState())

        graph.add_node("chat", self._chat_with_memory)
        graph.set_entry_point("chat")
        graph.add_edge("chat", END)

        self.app = graph.compile()

    async def _chat_with_memory(self, state: MemoryAgentState) -> MemoryAgentState:
        """Chat node with memory integration."""
        messages = convert_messages(system_prompts=[{"role": "system", "content": ""}], state=state)
        user_message = messages[-1]["content"]
        user_id = state.user_id

        # Retrieve relevant memories
        memories = []
        try:
            memory_results = self.memory.search(
                query=user_message,
                user_id=user_id,
                limit=3,
            )

            if "results" in memory_results:
                memories = [m["memory"] for m in memory_results["results"]]
            print(f"retreived {len(memories)} memories")
        except Exception as e:
            print(f"Memory retrieval error: {e}")

        # Build context
        memory_context = ""
        if memories:
            memory_context = f"\nRelevant memories:\n" + "\n".join([f"- {m}" for m in memories])

        # System prompt with memory
        system_prompt = f"""You are a helpful AI assistant with memory of past conversations.
        
{memory_context}

Be conversational, helpful, and reference past interactions when relevant."""

        # Convert messages
        messages = convert_messages(
            system_prompts=[{"role": "system", "content": system_prompt}], state=state
        )

        # Generate response
        response = await acompletion(model="gemini/gemini-2.0-flash", messages=messages)

        # Convert and store response
        ai_response = ModelResponseConverter(response, converter="litellm")
        # state.messages.append(Message.text_message(ai_response.message.content, role="assistant"))

        # Store interaction in memory
        try:
            interaction = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response.choices[0]["message"]["content"]},
            ]

            self.memory.add(messages=interaction, user_id=user_id, metadata={"app_id": self.app_id})
            print(f"✅ Memory stored for user {user_id}")
        except Exception as e:
            print(f"Memory storage error: {e}")

        return ai_response

    async def chat(self, message: str, user_id: str) -> str:
        """Simple chat interface."""
        # state = MemoryAgentState(
        #     messages=[Message.text_message(message, role="user")],
        #     user_id=user_id
        # )
        inp = {
            "messages": [Message.text_message(message, role="user")],
            "state": {"user_id": user_id},
        }
        config1 = {"thread_id": "12345", "recursion_limit": 10}
        result = await self.app.ainvoke(inp, config=config1)
        return result["messages"][-1].content


# Example usage
async def main():
    """Example conversation."""
    agent = SimplePersonalizedAgent()
    user_id = "test_user"

    # Conversation examples
    conversations = [
        "Hi, I'm John and I love pizza!",
        "What are some good pizza toppings?",
        "What do you remember about my food preferences?",
        "I also enjoy hiking on weekends",
        "What activities do I enjoy based on our conversation?",
    ]

    print("🤖 Simple Personalized Agent Demo\n")

    for i, message in enumerate(conversations, 1):
        print(f"👤 User: {message}")
        response = await agent.chat(message, user_id)
        print(f"🤖 Agent: {response}\n")

        if i < len(conversations):
            await asyncio.sleep(1)  # Brief pause


if __name__ == "__main__":
    asyncio.run(main())
