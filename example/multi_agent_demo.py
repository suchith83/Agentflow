"""
Multi-agent workflow demo using PyAgenity's Agent and AgentGraph.
"""

from pyagenity.agent.agent import Agent
from pyagenity.graph.base import AgentGraph


def main():
    # Define agents
    main_agent = Agent(name="main", model="google/gemini-pro")
    research_agent = Agent(name="research", model="google/gemini-pro")
    science_agent = Agent(name="science", model="google/gemini-pro")

    # Create graph and add agents
    graph = AgentGraph()
    graph.add_agent("main", main_agent)
    graph.add_agent("research", research_agent)
    graph.add_agent("science", science_agent)

    # Connect agents: main -> research -> science
    graph.connect("main", "research")
    graph.connect("research", "science")

    # Optional: custom router (route based on query content)
    def router(agent_name, query, state):
        if agent_name == "main":
            if "science" in query.lower():
                return "science"
            elif "search" in query.lower() or "find" in query.lower():
                return "research"
        elif agent_name == "research":
            if "science" in query.lower():
                return "science"
        return None

    graph.set_router(router)

    # Run a query
    query = "Can you find the latest science news?"
    results = graph.run(query)
    for agent, response in results.items():
        print(f"[{agent}] {getattr(response, 'content', response)}")


if __name__ == "__main__":
    main()
