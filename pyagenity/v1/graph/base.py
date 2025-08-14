# """
# Base classes and utilities for graph-based agents in PyAgenity.
# """

# from pyagenity.agent.agent import Agent


# class AgentGraph:
#     """
#     Multi-agent workflow engine for PyAgenity using custom Agent class.
#     Allows users to register agents, connect them, and route queries.
#     """

#     def __init__(self):
#         self.agents = {}
#         self.edges = []  # List of (src, dst)
#         self.entry_agent = None
#         self.routing_fn = None  # Optional: custom routing logic

#     def add_agent(self, name: str, agent: Agent):
#         self.agents[name] = agent
#         if self.entry_agent is None:
#             self.entry_agent = name

#     def connect(self, src: str, dst: str):
#         self.edges.append((src, dst))

#     def set_entry(self, name: str):
#         self.entry_agent = name

#     def set_router(self, fn):
#         """Set a custom routing function: fn(agent_name, query, state) -> next_agent_name"""
#         self.routing_fn = fn

#     def run(self, query, state=None):
#         """
#         Run the query through the agent graph, starting from entry_agent.
#         If routing_fn is set, use it to determine next agent.
#         Otherwise, follow edges in order.
#         """
#         if state is None:
#             state = {}
#         current = self.entry_agent
#         visited = set()
#         while current:
#             agent = self.agents[current]
#             response = agent.run(query)
#             state[current] = response
#             visited.add(current)
#             # Routing logic
#             next_agent = None
#             if self.routing_fn:
#                 next_agent = self.routing_fn(current, query, state)
#             else:
#                 # Default: follow first unvisited edge from current
#                 for src, dst in self.edges:
#                     if src == current and dst not in visited:
#                         next_agent = dst
#                         break
#             current = next_agent
#         return state
