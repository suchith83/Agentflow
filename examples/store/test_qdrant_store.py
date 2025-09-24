import asyncio
import sys
sys.path.append('examples/store')
from vector_memory_agent import VectorMemoryAgent

async def test_fixed_agent():
    try:
        agent = VectorMemoryAgent(use_cloud=True, embedding_dimensions=768)
        print('🤖 Testing fixed Gemini vector memory agent...')
        
        # Test first conversation
        response1 = await agent.chat(
            message='Hi! I am tharun. I am an ai engineer',
            user_id='tharun',
            conversation_id='session_1'
        )
        print(f'✅ First response: {response1}')
        
        # Test memory recall
        response2 = await agent.chat(
            message='this is tharun, What did I tell you about my work?',
            user_id='tharun',
            conversation_id='session_2'
        )
        print(f'✅ Memory recall: {response2}')
        # addign hobby
        response3 = await agent.chat(
            message='I like to play volleyball',
            user_id='tharun',
            conversation_id='session_3'
        )
        print(f'✅ Memory recall: {response3}')

        #recall hobby
        response4 = await agent.chat(
            message='what is my hobby',
            user_id='tharun',
            conversation_id='session_4'
        )
        print(f'✅ Memory recall: {response4}')

        response5 = await agent.chat(
            message='what game i like to play?',
            user_id='tharun',
            conversation_id='session_5'
        )
        print(f'✅ Memory recall: {response5}')
        
        
        await agent.cleanup()
        print('🎉 All tests completed successfully!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_fixed_agent())