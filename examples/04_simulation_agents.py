"""
Example demonstrating the simulation system with intelligent agents.
"""
import numpy as np
from graphfusionai.simulation import SimulationEnvironment, SimulationVisualizer
from graphfusionai.simulation.environment import TerrainType, Resource
from graphfusionai.simulation.agents import (
    ExplorerAgent,
    GathererAgent,
    CommunicatorAgent,
    DefenderAgent,
    LeaderAgent
)

def create_complex_terrain(env: SimulationEnvironment) -> None:
    """Create an interesting terrain layout."""
    # Create water body
    for x in range(5, 8):
        for y in range(5, 8):
            env.add_terrain(TerrainType.WATER, (x, y))
            
    # Create forest
    for x in range(12, 16):
        for y in range(12, 16):
            env.add_terrain(TerrainType.FOREST, (x, y))
            
    # Create mountain range
    for x, y in [(18, 18), (18, 19), (19, 18), (19, 19), (19, 20)]:
        env.add_terrain(TerrainType.MOUNTAIN, (x, y))
        
    # Create obstacles (rocks/walls)
    for x in range(8, 12):
        env.add_terrain(TerrainType.OBSTACLE, (x, 10))

def create_resources(env: SimulationEnvironment) -> None:
    """Create various resources in the environment."""
    # Food near water
    env.add_resource(Resource(
        type="fish",
        quantity=100,
        position=(5, 5),
        regeneration_rate=0.2,
        max_quantity=150
    ))
    
    env.add_resource(Resource(
        type="berries",
        quantity=80,
        position=(7, 7),
        regeneration_rate=0.15,
        max_quantity=100
    ))
    
    # Wood in forest
    env.add_resource(Resource(
        type="wood",
        quantity=200,
        position=(13, 13),
        regeneration_rate=0.05,
        max_quantity=300
    ))
    
    # Minerals in mountains
    env.add_resource(Resource(
        type="gold",
        quantity=150,
        position=(18, 18),
        regeneration_rate=0.02,
        max_quantity=200
    ))
    
    env.add_resource(Resource(
        type="iron",
        quantity=180,
        position=(19, 19),
        regeneration_rate=0.03,
        max_quantity=250
    ))

def create_agent_team(env: SimulationEnvironment) -> None:
    """Create a team of specialized agents."""
    # Create leader at center
    leader = LeaderAgent("Leader-1", (10, 10))
    env.add_agent(leader)
    
    # Create explorers at corners
    explorer_positions = [(0, 0), (0, 19), (19, 0)]
    for i, pos in enumerate(explorer_positions):
        explorer = ExplorerAgent(f"Explorer-{i+1}", pos)
        env.add_agent(explorer)
        
    # Create gatherers near resources
    gatherer_positions = [(4, 4), (12, 12), (17, 17)]
    for i, pos in enumerate(gatherer_positions):
        gatherer = GathererAgent(f"Gatherer-{i+1}", pos)
        env.add_agent(gatherer)
        
    # Create communicators at strategic points
    comm_positions = [(7, 7), (13, 13)]
    for i, pos in enumerate(comm_positions):
        communicator = CommunicatorAgent(f"Comm-{i+1}", pos)
        env.add_agent(communicator)
        
    # Create defenders near valuable resources
    defender_positions = [(18, 17), (17, 18)]
    for i, pos in enumerate(defender_positions):
        defender = DefenderAgent(f"Defender-{i+1}", pos)
        env.add_agent(defender)

def main():
    """Run simulation example."""
    print("Initializing simulation environment...")
    
    # Create environment
    env = SimulationEnvironment("Complex Simulation", size=(20, 20))
    
    # Setup environment
    print("Creating terrain...")
    create_complex_terrain(env)
    
    print("Adding resources...")
    create_resources(env)
    
    print("Creating agent team...")
    create_agent_team(env)
    
    # Create visualizer
    vis = SimulationVisualizer(env)
    
    print("\nSimulation ready!")
    print("Environment size:", env.size)
    print("Number of agents:", len(env.agents))
    print("Number of resources:", len(env.resources))
    
    print("\nStarting simulation visualization...")
    print("- Close the visualization window to end simulation")
    print("- Resources are shown as yellow circles")
    print("- Agents are shown as red dots with labels")
    print("- Communication links are shown as gray lines")
    print("- Different terrain types have different colors")
    
    # Run simulation with visualization
    try:
        vis.start_animation(interval=100)  # Update every 100ms
        
        # After simulation ends, show metrics
        print("\nSimulation ended")
        print("Showing final metrics...")
        vis.visualize_metrics()
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        
if __name__ == "__main__":
    main()
