# """
# Multi-agent workflow demo using PyAgenity's Agent and AgentGraph.
# """

# from pyagenity.graph.graph import AgentGraph


# def build_agent(name, model):
#     return [{"role": "user", "content": f"Create a {name} agent with model {model}."}]


# def main():
#     # Define agents
#     main_agent = build_agent(name="main", model="google/gemini-pro")
#     research_agent = build_agent(name="research", model="google/gemini-pro")
#     science_agent = build_agent(name="science", model="google/gemini-pro")

#     # Create graph and add agents
#     graph = AgentGraph()
#     graph.add_node("main", main_agent)
#     graph.add_node("research", research_agent)
#     graph.add_node("science", science_agent)

#     # Connect agents: main -> research -> science
#     graph.connect("main", "research")
#     graph.connect("research", "science")

#     # Optional: custom router (route based on query content)
#     def router(agent_name, query, state):
#         if agent_name == "main":
#             if "science" in query.lower():
#                 return "science"
#             elif "search" in query.lower() or "find" in query.lower():
#                 return "research"
#         elif agent_name == "research":
#             if "science" in query.lower():
#                 return "science"
#         return None

#     graph.set_router(router)

#     # Run a query
#     query = "Can you find the latest science news?"
#     results = graph.run(query)
#     for agent, response in results.items():
#         print(f"[{agent}] {getattr(response, 'content', response)}")


# if __name__ == "__main__":
#     main()
