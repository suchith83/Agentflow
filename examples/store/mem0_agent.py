"""
Simple Personalized Agent Example using PyAgenity Mem0Store

A refactored version of the simple personalized agent that uses PyAgenity's
Mem0Store instead of direct Mem0 integration. This demonstrates:

- PyAgenity StateGraph with memory integration
- Mem0Store for standardized memory operations
- Cloud Qdrant vector storage via Mem0Store
- Message-based memory storage and retrieval
- Better error handling and abstraction

This shows how to migrate from direct Mem0 usage to PyAgenity's store framework.
"""

import os
import asyncio
from typing import Dict, Any

from dotenv import load_dotenv
from litellm import acompletion

from pyagenity.graph import StateGraph
from pyagenity.state.agent_state import AgentState
from pyagenity.store.mem0_store import create_mem0_store_with_qdrant
from pyagenity.store.store_schema import MemoryType
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages
from pyagenity.adapters.llm.model_response_converter import ModelResponseConverter

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")


class MemoryAgentState(AgentState):
    """State with user ID for memory context."""

    user_id: str = ""


class SimplePersonalizedAgentWithStore:
    """
    Simple personalized agent using PyAgenity Mem0Store for memory.

    This version demonstrates how to use the PyAgenity store framework
    instead of direct Mem0 integration for better abstraction and
    standardized memory operations.
    """

    def __init__(self):
        # Initialize Mem0Store with Qdrant configuration
        self.store = create_mem0_store_with_qdrant(
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="simple_agent_memory_store",
            embedding_model="models/text-embedding-004",  # Gemini embeddings
            llm_model="gemini-2.0-flash-exp",  # Gemini LLM
            app_id="simple-agent-store",
            # Specify providers correctly
            embedder_provider="gemini",
            llm_provider="gemini",
            # Additional config for Gemini/Qdrant integration
            vector_store_config={
                "embedding_model_dims": 768  # Gemini embedding dimensions
            },
            llm_config={"temperature": 0.1},
        )

        self._build_graph()

    def _build_graph(self):
        """Build PyAgenity graph with memory integration."""
        graph = StateGraph[MemoryAgentState](MemoryAgentState())

        graph.add_node("chat_with_store", self._chat_with_store_memory)
        graph.set_entry_point("chat_with_store")
        graph.add_edge("chat_with_store", END)

        self.app = graph.compile()

    async def _chat_with_store_memory(self, state: MemoryAgentState) -> MemoryAgentState:
        """Chat node using PyAgenity Mem0Store for memory operations."""
        messages = convert_messages([], state)
        user_message = messages[-1]["content"]
        user_id = state.user_id

        # Retrieve relevant memories using store's search method
        memory_context = ""
        try:
            # Use store's search method with better error handling
            config = {"user_id": user_id, "thread_id": f"chat_{user_id}"}
            memory_results = await self.store.asearch(
                config,
                query=user_message,
                limit=3,
                score_threshold=0.5,  # Only get relevant memories
                memory_type=MemoryType.EPISODIC,  # Filter by memory type (now using episodic for compatibility)
            )

            if memory_results:
                memories = [result.content for result in memory_results]
                memory_context = f"\nRelevant memories from past conversations:\n" + "\n".join(
                    [f"- {memory}" for memory in memories]
                )
                print(f"📚 Retrieved {len(memories)} relevant memories")
            else:
                print("📚 No relevant memories found")

        except Exception as e:
            print(f"❌ Memory retrieval error: {e}")
            # Continue without memories if retrieval fails

        # Build enhanced system prompt with memory context
        system_prompt = f"""You are a helpful AI assistant with memory of past conversations.

{memory_context}

Be conversational, helpful, and reference past interactions when relevant. 
Show that you remember previous topics and user preferences."""

        # Convert messages for LLM
        messages = convert_messages(
            system_prompts=[{"role": "system", "content": system_prompt}], state=state
        )

        # Generate response using LiteLLM
        # try:
        response = await acompletion(
            model="gemini/gemini-2.0-flash", messages=messages, temperature=0.7
        )

        assistant_content = response.choices[0].message.content

        # Convert response to PyAgenity Message and create return state
        assistant_message = Message.text_message(assistant_content, role="assistant")

        # Store the conversation in memory using store's message storage
        try:
            config = {"user_id": user_id, "thread_id": f"chat_{user_id}"}

            # Store user message
            user_msg = Message.text_message(user_message, role="user")
            await self.store.astore(
                config,
                content=user_msg,
                memory_type=MemoryType.EPISODIC,  # Use episodic for compatibility
                category="chat",
                metadata={"session_id": "main_chat", "interaction_type": "user_input"},
            )

            # Store assistant response
            assistant_msg = Message.text_message(assistant_content, role="assistant")
            await self.store.astore(
                config,
                content=assistant_msg,
                memory_type=MemoryType.EPISODIC,  # Use episodic for compatibility
                category="chat",
                metadata={
                    "session_id": "main_chat",
                    "interaction_type": "assistant_response",
                },
            )

            print(f"💾 Stored conversation for user {user_id}")

        except Exception as e:
            print(f"❌ Memory storage error: {e}")
            # Continue even if storage fails

        # Return updated state with new message
        return MemoryAgentState(context=[*state.context, assistant_message], user_id=state.user_id)

    async def chat(self, message: str, user_id: str) -> str:
        """Simple chat interface using PyAgenity store."""
        try:
            # Create initial state with proper structure
            initial_state = MemoryAgentState(
                context=[Message.text_message(message, role="user")], user_id=user_id
            )

            config = {"thread_id": f"chat_{user_id}", "recursion_limit": 10}

            # Invoke the graph - pass as dictionary
            result = await self.app.ainvoke(initial_state.model_dump(), config=config)
            # Result should be a dictionary with context
            return result["context"][-1].content

        except Exception as e:
            print(f"❌ Chat error: {e}")
            return "I apologize, but I encountered an error processing your message."

    async def get_memory_stats(self, user_id: str) -> dict[str, Any]:
        """Get memory statistics using store's get_all method."""
        try:
            config = {"user_id": user_id, "thread_id": f"stats_{user_id}"}
            all_memories = await self.store.aget_all(config)

            stats = {
                "user_id": user_id,
                "total_memories": len(all_memories),
                "memory_types": {},
                "categories": {},
            }

            for memory in all_memories:
                # Count memory types
                memory_type = memory.memory_type.value if memory.memory_type else "unknown"
                stats["memory_types"][memory_type] = stats["memory_types"].get(memory_type, 0) + 1

                # Count categories
                category = (
                    memory.metadata.get("category", "general") if memory.metadata else "general"
                )
                stats["categories"][category] = stats["categories"].get(category, 0) + 1

            return stats
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {"error": str(e)}

    async def clear_user_memories(self, user_id: str) -> int:
        """Clear all memories for a user using store's forget_memory method."""
        try:
            config = {"user_id": user_id, "thread_id": f"clear_{user_id}"}
            result = await self.store.aforget_memory(config)
            deleted_count = result.get("deleted_count", 0) if isinstance(result, dict) else 0
            print(f"🗑️ Deleted {deleted_count} memories for user {user_id}")
            return deleted_count
        except Exception as e:
            print(f"❌ Delete error: {e}")
            return 0

    async def search_memories(self, query: str, user_id: str, limit: int = 5) -> list:
        """Search memories using store's search method."""
        try:
            config = {"user_id": user_id, "thread_id": f"search_{user_id}"}
            results = await self.store.asearch(
                config,
                query=query,
                limit=limit,
                memory_type=MemoryType.EPISODIC,  # Use episodic for compatibility
            )

            return [
                {
                    "content": result.content,
                    "score": result.score,
                    "created_at": result.timestamp.isoformat() if result.timestamp else None,
                    "metadata": result.metadata,
                }
                for result in results
            ]
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []


# Example usage with enhanced functionality
async def main():
    """Enhanced example conversation with store features."""
    agent = SimplePersonalizedAgentWithStore()
    user_id = "jack"

    print("🤖 Simple Personalized Agent with PyAgenity Mem0Store\n")
    print("Features:")
    print("- Memory storage via PyAgenity Mem0Store")
    print("- Standardized memory operations")
    print("- Better error handling and abstraction")
    print("- Memory statistics and management\n")

    # Initial conversation
    conversations = [
        "Hi, I'm Alice and I love cooking Italian food!",
        "What are some good pasta recipes?",
        "I also enjoy gardening in my spare time",
        "What do you remember about my hobbies?",
        "Can you suggest activities that combine my interests?",
    ]

    for i, message in enumerate(conversations, 1):
        print(f"👤 User: {message}")
        response = await agent.chat(message, user_id)
        print(f"🤖 Agent: {response}\n")

        if i < len(conversations):
            await asyncio.sleep(1)  # Brief pause

    print("=" * 60)
    print("📊 MEMORY STATISTICS")
    print("=" * 60)

    # Show memory statistics
    stats = await agent.get_memory_stats(user_id)
    print(f"User: {stats.get('user_id')}")
    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"Memory types: {stats.get('memory_types', {})}")
    print(f"Categories: {stats.get('categories', {})}")
    print()

    print("=" * 60)
    print("🔍 MEMORY SEARCH")
    print("=" * 60)

    # Demonstrate memory search
    search_query = "food preferences"
    search_results = await agent.search_memories(search_query, user_id, limit=3)
    print(f"Search query: '{search_query}'")
    print(f"Found {len(search_results)} results:")

    for i, result in enumerate(search_results, 1):
        print(f"{i}. Content: {result['content'][:100]}...")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Created: {result['created_at']}")
        print()

    print("=" * 60)
    print("🧹 CLEANUP DEMO")
    print("=" * 60)

    # Demonstrate cleanup (commented out to preserve demo data)
    # deleted_count = await agent.clear_user_memories(user_id)
    # print(f"Deleted {deleted_count} memories")

    print("Demo completed! Memory persists for future conversations.")


if __name__ == "__main__":
    asyncio.run(main())
