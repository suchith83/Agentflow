We have class called Agent.

We will create one base class for Agent.
Class BaseAgent:
    - common methods and attributes for all agents

Then we have Agent will be modified to inherit from BaseAgent.
Same Way we will expose TestAgent which will also inherit from BaseAgent.
So easily we can swap between Agent and TestAgent in tests.


Now lets handle Functions...

Current:
# This already works - node functions can be simple
async def my_node(state: AgentState, config: dict):
    return [Message.text_message("Hello")]

graph = StateGraph()
graph.add_node("MAIN", my_node)  # ✅ Simple and direct


Proposed:
# Production
async def my_node(state: AgentState, config: dict):
    return [Message.text_message("Hello")]

graph = StateGraph()
graph.add_node("MAIN", my_node)  # ✅ Simple and direct

Testing:
# This already works - node functions can be simple
async def my_node_test(state: AgentState, config: dict):
    return [Message.text_message("Hello")]

graph = StateGraph()
graph.override_node("MAIN", my_node_test)  # ✅ Simple and direct

This way its very simple, user side no need to create complex mocking functions or too much abstractions or changes in code.


# Now we need to think about TestNode, what is the idea?

'Node` class and `Edge` class not exposed so maybe no need to mock them, what do you say?

Now Inside InvokeHandler how can we get the nodes from INjectQ container, insted list of the graph nodes?