"""
Demonstration of the simulation system.
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

def create_terrain(env: SimulationEnvironment) -> None:
    """Create interesting terrain."""
    # Add some water
    for x, y in [(5, 5), (5, 6), (6, 5), (6, 6)]:
        env.add_terrain(TerrainType.WATER, (x, y))
        
    # Add forest
    for x in range(12, 15):
        for y in range(12, 15):
            env.add_terrain(TerrainType.FOREST, (x, y))
            
    # Add mountains
    for x, y in [(18, 18), (18, 19), (19, 18), (19, 19)]:
        env.add_terrain(TerrainType.MOUNTAIN, (x, y))
        
    # Add obstacles
    for x in range(8, 11):
        env.add_terrain(TerrainType.OBSTACLE, (x, 10))

def create_resources(env: SimulationEnvironment) -> None:
    """Create resources."""
    # Food near water
    env.add_resource(Resource(
        type="food",
        quantity=100,
        position=(4, 5),
        regeneration_rate=0.1,
        max_quantity=150
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
        type="minerals",
        quantity=150,
        position=(18, 18),
        regeneration_rate=0.02,
        max_quantity=200
    ))

def create_agents(env: SimulationEnvironment) -> None:
    """Create agents."""
    # Create leader
    leader = LeaderAgent("Leader-1", (10, 10))
    env.add_agent(leader)
    
    # Create explorers
    for i in range(2):
        explorer = ExplorerAgent(
            f"Explorer-{i+1}",
            (np.random.randint(0, env.size[0]),
             np.random.randint(0, env.size[1]))
        )
        env.add_agent(explorer)
        
    # Create gatherers
    for i in range(3):
        gatherer = GathererAgent(
            f"Gatherer-{i+1}",
            (np.random.randint(0, env.size[0]),
             np.random.randint(0, env.size[1]))
        )
        env.add_agent(gatherer)
        
    # Create communicators
    for i in range(2):
        communicator = CommunicatorAgent(
            f"Comm-{i+1}",
            (np.random.randint(0, env.size[0]),
             np.random.randint(0, env.size[1]))
        )
        env.add_agent(communicator)
        
    # Create defenders
    for i in range(2):
        defender = DefenderAgent(
            f"Defender-{i+1}",
            (np.random.randint(0, env.size[0]),
             np.random.randint(0, env.size[1]))
        )
        env.add_agent(defender)

def main():
    """Run simulation demo."""
    # Create environment
    env = SimulationEnvironment("Demo Simulation", size=(20, 20))
    
    # Setup environment
    create_terrain(env)
    create_resources(env)
    create_agents(env)
    
    # Create visualizer
    vis = SimulationVisualizer(env)
    
    # Start animation
    print("Starting simulation...")
    print("- Close the visualization window to end simulation")
    print("- Resources are shown as yellow circles")
    print("- Agents are shown as red dots with labels")
    print("- Communication links are shown as gray lines")
    print("- Terrain types have different colors (see legend)")
    vis.start_animation(interval=100)  # Update every 100ms
    
    # After animation ends, show final metrics
    print("\nSimulation ended")
    print("Showing final metrics...")
    vis.visualize_metrics()
    
if __name__ == "__main__":
    main()
