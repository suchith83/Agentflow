"""
Vector Memory Agent Example using PyAgenity + QdrantVectorStore + Gemini Models

This example demonstrates how to use PyAgenity's QdrantVectorStore for long-term memory
in conversational agents with Google's Gemini models for both chat and embeddings. The agent can:

1. Store conversation messages as vectors using Gemini embeddings (text-embedding-004)
2. Retrieve relevant past conversations based on semantic similarity  
3. Maintain context across multiple conversation sessions using Gemini chat models
4. Use both local and cloud Qdrant deployments
5. Configure embedding dimensions (128-3072, recommended: 768, 1536, 3072)

Features demonstrated:
- QdrantVectorStore integration with PyAgenity
- Gemini text-embedding-004 for creating embeddings
- Gemini 1.5 Flash for chat completion
- Message storage and retrieval with configurable embedding dimensions
- Semantic similarity search across conversation history
- Multi-user conversation management
- Vector store statistics and management

Required Environment Variables:
- GEMINI_API_KEY: Your Google AI/Gemini API key
- QDRANT_URL: (optional) Cloud Qdrant URL
- QDRANT_API_KEY: (optional) Cloud Qdrant API key
"""

import os
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from litellm import acompletion, aembedding
from pydantic import Field

from pyagenity.graph import StateGraph
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages
from pyagenity.adapters.llm.model_response_converter import ModelResponseConverter
from pyagenity.store.qdrant_vector_store import (
    QdrantVectorStore, 
    create_cloud_qdrant_vector_store,
    create_local_qdrant_vector_store
)
from pyagenity.store.vector_base_store import DistanceMetric

# Load environment variables
load_dotenv()

class VectorMemoryState(AgentState):
    """Extended state for vector memory agent."""
    user_id: str = ""
    conversation_id: str = ""
    context_retrieved: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorMemoryAgent:
    """
    Conversational agent with vector-based long-term memory using QdrantVectorStore.
    
    This agent demonstrates:
    - Storing conversations as vector embeddings
    - Retrieving contextually relevant past conversations
    - Managing multiple users and conversation sessions
    - Using semantic search for memory recall
    """
    
    def __init__(self, use_cloud: bool = True, embedding_dimensions: int = 768):
        """
        Initialize the vector memory agent.
        
        Args:
            use_cloud: Whether to use cloud Qdrant or local storage
            embedding_dimensions: Dimensions for Gemini embeddings (128, 256, 512, 768, 1536, 2048, 3072)
        """
        self.use_cloud = use_cloud
        self.collection_name = "conversation_memory"
        self.embedding_model = "gemini/text-embedding-004"  # Gemini embeddings
        self.chat_model = "gemini/gemini-1.5-flash"  # Gemini chat model
        self.embedding_dimensions = embedding_dimensions  # Configurable dimensions
        
        # Initialize vector store
        self._setup_vector_store()
        
        # Build PyAgenity graph
        self._build_graph()
        
        # Collection will be created on first use
    
    def _setup_vector_store(self):
        """Setup QdrantVectorStore based on configuration."""
        if self.use_cloud:
            # Cloud Qdrant configuration
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            
            if not qdrant_url or not qdrant_api_key:
                raise ValueError("QDRANT_URL and QDRANT_API_KEY required for cloud mode")
            
            self.vector_store = create_cloud_qdrant_vector_store(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            print("🌐 Using cloud Qdrant vector store")
        else:
            # Local Qdrant configuration
            self.vector_store = create_local_qdrant_vector_store(
                path="./qdrant_data"
            )
            print("💾 Using local Qdrant vector store")
    
    async def _ensure_collection_exists(self):
        """Ensure the conversation collection exists."""
        try:
            collection_exists = await self.vector_store.acollection_exists(self.collection_name)
            
            if not collection_exists:
                await self.vector_store.acreate_collection(
                    name=self.collection_name,
                    vector_size=self.embedding_dimensions,  # Use configurable dimensions
                    distance_metric=DistanceMetric.COSINE
                )
                print(f"✅ Created collection: {self.collection_name} with {self.embedding_dimensions}D vectors")
            else:
                print(f"📁 Collection already exists: {self.collection_name}")
                
            # Always try to create index for user_id filtering (idempotent operation)  
            # Note: Index creation not supported yet, using application-level filtering
            print("� Using application-level filtering for user_id (index creation not available)")
                
            # Display collection stats (handle different response formats)
            try:
                stats = await self.vector_store.aget_collection_stats(self.collection_name)
                # Handle different possible response formats
                if isinstance(stats, dict):
                    vector_count = stats.get('vectors_count', 0)
                    point_count = stats.get('points_count', 0)
                else:
                    vector_count = getattr(stats, 'vectors_count', 0)
                    point_count = getattr(stats, 'points_count', 0)
                print(f"📊 Collection stats: {vector_count + point_count} vectors and points stored")
            except Exception as stats_error:
                print(f"⚠️ Could not retrieve collection stats: {stats_error}")
                
        except Exception as e:
            print(f"❌ Error setting up collection: {e}")
    
    def _build_graph(self):
        """Build the PyAgenity conversation graph."""
        graph = StateGraph[VectorMemoryState](VectorMemoryState())
        
        # Add nodes
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("store_memory", self._store_memory)
        
        # Define flow
        graph.set_entry_point("retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
        graph.add_edge("generate_response", "store_memory")
        graph.add_edge("store_memory", END)
        
        self.app = graph.compile()
    
    async def _retrieve_context(self, state: VectorMemoryState) -> VectorMemoryState:
        """Retrieve relevant conversation context from vector store."""
        user_message = state.context[-1].text()
        
        # Ensure collection exists before first use
        await self._ensure_collection_exists()
        
        try:
            # Generate embedding for the user message
            embedding_response = await aembedding(
                model=self.embedding_model,
                input=[user_message],
                dimensions=self.embedding_dimensions  # Set Gemini embedding dimensions
            )
            query_vector = embedding_response.data[0].embedding
            
            # Search for similar conversations (no user filtering due to index issue)
            similar_memories = await self.vector_store.asearch(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=5,
                # filters={"user_id": state.user_id},
                score_threshold=0.5
            )
            
            # Extract relevant context (filter by user_id at application level)
            context_messages = []
            for memory in similar_memories:
                # Filter by user_id at application level since index filtering failed
                if (memory.score > 0.5 and 
                    memory.payload.get("user_id") == state.user_id):  # High similarity threshold
                    context_messages.append({
                        "content": memory.payload.get("content", ""),
                        "role": memory.payload.get("role", ""),
                        "timestamp": memory.payload.get("timestamp", ""),
                        "similarity": memory.score
                    })
            
            # Add context to metadata
            if context_messages:
                state.metadata["retrieved_context"] = context_messages
                state.context_retrieved = True
                print(f"🔍 Retrieved {len(context_messages)} relevant memories")
            else:
                print("🆕 No relevant context found - starting fresh conversation")
            
        except Exception as e:
            print(f"⚠️ Context retrieval error: {e}")
            state.context_retrieved = False
        
        return state
    
    async def _generate_response(self, state: VectorMemoryState) -> VectorMemoryState:
        """Generate response using retrieved context."""
        
        # Build context from retrieved memories
        context_text = ""
        if state.context_retrieved and "retrieved_context" in state.metadata:
            context_memories = state.metadata["retrieved_context"]
            if context_memories:
                context_text = "\n--- Relevant conversation history ---\n"
                for memory in context_memories[:3]:  # Use top 3 most similar
                    role_emoji = "👤" if memory["role"] == "user" else "🤖"
                    context_text += f"{role_emoji} {memory['content']}\n"
                context_text += "--- End of history ---\n\n"
        
        # System prompt with context
        system_prompt = f"""You are a helpful AI assistant with access to conversation history.
        
{context_text}Use the conversation history to provide contextually relevant responses.
Reference past conversations when appropriate, but don't be overly repetitive.
Be natural and conversational.

User ID: {state.user_id}
Conversation ID: {state.conversation_id}"""
        
        # Generate response
        try:
            messages = convert_messages(
                system_prompts=[{"role": "system", "content": system_prompt}],
                state=state
            )
            
            response = await acompletion(
                model=self.chat_model,
                messages=messages,
                temperature=0.7
            )
            
            # Convert and add response to state
            ai_response = ModelResponseConverter(response, converter="litellm")
            response_text = response.choices[0].message.content
            
            # state.context.append(
            #     Message.text_message(response_text, role="assistant")
            # )
            
            print(f"🤖 Generated response using {len(context_text)} chars of context")
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            # state.context.append(
            #     Message.text_message(error_msg, role="assistant")
            # )
            print(f"❌ Response generation error: {e}")
        
        return ai_response
    
    async def _store_memory(self, state: VectorMemoryState) -> VectorMemoryState:
        """Store the conversation in vector memory."""
        try:
            # Store both user message and assistant response
            messages_to_store = state.context[-2:]  # Last 2 messages (user + assistant)
            
            for message in messages_to_store:
                content = message.text()
                if not content.strip():
                    continue
                
                # Generate embedding
                embedding_response = await aembedding(
                    model=self.embedding_model,
                    input=[content],
                    dimensions=self.embedding_dimensions  # Set Gemini embedding dimensions
                )
                vector = embedding_response.data[0].embedding
                
                # Prepare metadata
                payload = {
                    "content": content,
                    "role": message.role,
                    "user_id": state.user_id,
                    "conversation_id": state.conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "message_id": str(message.message_id)
                }
                
                # Store in vector database with a valid UUID
                point_id = str(uuid.uuid4())
                await self.vector_store.ainsert(
                    collection_name=self.collection_name,
                    vectors=vector,
                    payloads=payload,
                    ids=point_id
                )
            
            print(f"💾 Stored {len(messages_to_store)} messages in vector memory")
            
        except Exception as e:
            print(f"⚠️ Memory storage error: {e}")
        
        return state
    
    async def chat(
        self, 
        message: str, 
        user_id: str, 
        conversation_id: str = "default"
    ) -> str:
        """
        Main chat interface.
        
        Args:
            message: User message
            user_id: Unique user identifier
            conversation_id: Conversation session ID
            
        Returns:
            Assistant response
        """
        # Create input dictionary following PyAgenity pattern
        input_dict = {
            "messages": [Message.text_message(message, role="user")],
            "state": {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "context_retrieved": False,
                "metadata": {},
            }
        }
        
        # Process through graph
        config = {
            "thread_id": f"{user_id}_{conversation_id}",
            "recursion_limit": 10
        }
        
        result = await self.app.ainvoke(input_dict, config=config)
        return result["messages"][-1].content
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            stats = await self.vector_store.aget_collection_stats(self.collection_name)
            collections = await self.vector_store.alist_collections()
            
            return {
                "total_collections": len(collections),
                "conversation_collection": stats,
                "all_collections": collections
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def search_memories(
        self, 
        query: str, 
        user_id: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.
        
        Args:
            query: Search query
            user_id: Optional user filter
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        try:
            # Generate query embedding
            embedding_response = await aembedding(
                model=self.embedding_model,
                input=[query],
                dimensions=self.embedding_dimensions  # Set Gemini embedding dimensions
            )
            query_vector = embedding_response.data[0].embedding
            
            # Build filters
            filters = {}
            if user_id:
                filters["user_id"] = user_id
            
            # Search
            results = await self.vector_store.asearch(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                filters=filters if filters else None,
                score_threshold=0.5
            )
            
            # Format results
            memories = []
            for result in results:
                memories.append({
                    "content": result.payload.get("content", ""),
                    "role": result.payload.get("role", ""),
                    "user_id": result.payload.get("user_id", ""),
                    "timestamp": result.payload.get("timestamp", ""),
                    "similarity_score": result.score,
                    "id": result.id
                })
            
            return memories
            
        except Exception as e:
            print(f"❌ Memory search error: {e}")
            return []
    
    async def cleanup(self):
        """Clean up vector store resources."""
        await self.vector_store.acleanup()
        print("🧹 Vector store cleanup completed")


async def demo_conversation():
    """Demonstration of the vector memory agent."""
    print("🚀 Vector Memory Agent Demo (Using Gemini Models)")
    print("=" * 60)
    
    # Initialize agent (try cloud first, fallback to local)
    try:
        agent = VectorMemoryAgent(use_cloud=True, embedding_dimensions=768)
        print("☁️ Using cloud Qdrant with Gemini models")
    except:
        print("⚠️ Cloud Qdrant not available, using local storage with Gemini models")
        agent = VectorMemoryAgent(use_cloud=False, embedding_dimensions=768)
    
    await asyncio.sleep(1)  # Allow collection setup
    
    # Demo conversation with user Alice
    user_id = "alice"
    conv_id = "session_1"
    
    print(f"\n💬 Starting conversation with user: {user_id}")
    
    # Multi-turn conversation
    messages = [
        "Hi! I'm Alice and I work as a data scientist.",
        "I'm working on a machine learning project with vector databases.",
        "Can you help me understand similarity search algorithms?",
        "What are the advantages of cosine similarity vs euclidean distance?"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 Alice: {message}")
        
        response = await agent.chat(message, user_id, conv_id)
        print(f"🤖 Agent: {response}")
        
        # Brief pause between messages
        await asyncio.sleep(1)
    
    # Start new conversation session to test memory retrieval
    print(f"\n🔄 Starting new conversation session...")
    conv_id = "session_2"
    
    new_messages = [
        "Hi again! Do you remember what I do for work?",
        "Can you remind me about the project I mentioned earlier?",
        "What similarity metrics did we discuss before?"
    ]
    
    for i, message in enumerate(new_messages, 1):
        print(f"\n--- New Session Turn {i} ---")
        print(f"👤 Alice: {message}")
        
        response = await agent.chat(message, user_id, conv_id)
        print(f"🤖 Agent: {response}")
        
        await asyncio.sleep(1)
    
    # Demonstrate memory search
    print(f"\n🔍 Searching memories for 'machine learning'...")
    memories = await agent.search_memories("machine learning", user_id="alice")
    
    print(f"Found {len(memories)} relevant memories:")
    for i, memory in enumerate(memories[:3], 1):
        print(f"{i}. [{memory['role']}] {memory['content'][:100]}... "
              f"(similarity: {memory['similarity_score']:.3f})")
    
    # Show statistics
    print(f"\n📊 Memory Statistics:")
    stats = await agent.get_memory_stats()
    if "error" not in stats:
        collection_stats = stats["conversation_collection"]
        print(f"Total vectors: {collection_stats.get('vectors_count', 0)}")
        print(f"Collections: {stats['total_collections']}")
    
    # Cleanup
    await agent.cleanup()
    print("\n✅ Demo completed successfully!")


async def interactive_mode():
    """Interactive chat mode for testing."""
    print("🎯 Interactive Vector Memory Agent (Using Gemini Models)")
    print("Type 'quit' to exit, 'stats' for statistics, 'search <query>' to search memories")
    print("=" * 70)
    
    # Initialize agent
    try:
        agent = VectorMemoryAgent(use_cloud=True, embedding_dimensions=768)
    except:
        agent = VectorMemoryAgent(use_cloud=False, embedding_dimensions=768)
    
    await asyncio.sleep(1)
    
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    conv_id = "interactive_session"
    print(f"\n💬 Chatting as user: {user_id}")
    
    while True:
        try:
            message = input(f"\n👤 {user_id}: ").strip()
            
            if message.lower() in ['quit', 'exit', 'bye']:
                break
            elif message.lower() == 'stats':
                stats = await agent.get_memory_stats()
                print(f"📊 Stats: {json.dumps(stats, indent=2)}")
                continue
            elif message.lower().startswith('search '):
                query = message[7:].strip()
                memories = await agent.search_memories(query, user_id)
                print(f"🔍 Found {len(memories)} memories for '{query}':")
                for mem in memories[:5]:
                    print(f"  - {mem['content'][:80]}... (score: {mem['similarity_score']:.3f})")
                continue
            elif not message:
                continue
            
            response = await agent.chat(message, user_id, conv_id)
            print(f"🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    await agent.cleanup()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(demo_conversation())