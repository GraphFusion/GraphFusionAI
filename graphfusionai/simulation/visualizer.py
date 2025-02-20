"""
Enhanced simulation visualizer with advanced features.
"""
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from .environment import TerrainType, SimulationEnvironment

class SimulationVisualizer:
    """Enhanced simulation visualizer with advanced features."""

    def __init__(self, environment: SimulationEnvironment):
        """
        Initialize visualizer.
        
        Args:
            environment: Simulation environment to visualize
        """
        self.environment = environment
        self.fig = None
        self.ax = None
        self.animation = None
        self.paused = False
        
        # Color maps
        self.terrain_colors = {
            TerrainType.EMPTY: 'white',
            TerrainType.OBSTACLE: 'gray',
            TerrainType.WATER: 'blue',
            TerrainType.FOREST: 'green',
            TerrainType.MOUNTAIN: 'brown',
            TerrainType.RESOURCE: 'gold'
        }
        
    def setup_plot(self) -> None:
        """Set up the plot for visualization."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-0.5, self.environment.size[0] - 0.5)
        self.ax.set_ylim(-0.5, self.environment.size[1] - 0.5)
        self.ax.grid(True)
        self.ax.set_title(f"Simulation: {self.environment.name}")
        
    def visualize_terrain(self) -> None:
        """Visualize terrain."""
        terrain_grid = np.array([[t.value for t in row] for row in self.environment.terrain])
        plt.imshow(
            terrain_grid,
            cmap='terrain',
            extent=(-0.5, self.environment.size[0] - 0.5, -0.5, self.environment.size[1] - 0.5)
        )
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=t.name)
            for t, color in self.terrain_colors.items()
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
    def visualize_resources(self) -> None:
        """Visualize resources."""
        for resource in self.environment.resources:
            x, y = resource.position
            size = np.sqrt(resource.quantity) * 100  # Scale size by quantity
            plt.scatter(x, y, s=size, c='yellow', alpha=0.5, label=f"{resource.type} ({resource.quantity:.1f})")
            
    def visualize_agents(self) -> None:
        """Visualize agents."""
        agent_positions = [agent.position for agent in self.environment.agents]
        if not agent_positions:
            return
            
        x, y = zip(*agent_positions)
        plt.scatter(x, y, c='red', s=100, label='Agents')
        
        # Add agent labels
        for agent in self.environment.agents:
            plt.annotate(
                agent.name,
                agent.position,
                xytext=(5, 5),
                textcoords='offset points'
            )
            
    def visualize_communication_network(self) -> None:
        """Visualize agent communication network."""
        G = nx.Graph()
        
        # Add nodes
        for agent in self.environment.agents:
            G.add_node(agent.name, pos=agent.position)
            
        # Add edges
        for agent1, connected in self.environment.communication_links.items():
            for agent2 in connected:
                G.add_edge(agent1.name, agent2.name)
                
        # Draw network
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos)
        
    def visualize_metrics(self) -> None:
        """Visualize simulation metrics."""
        metrics = self.environment.get_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Simulation Metrics')
        
        # Agent count
        axes[0, 0].plot(metrics['agent_counts'])
        axes[0, 0].set_title('Agent Count')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Count')
        
        # Resource levels
        resource_levels = np.array(metrics['resource_levels'])
        if resource_levels.size > 0:
            for i in range(resource_levels.shape[1]):
                axes[0, 1].plot(resource_levels[:, i], label=f'Resource {i+1}')
            axes[0, 1].set_title('Resource Levels')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Quantity')
            axes[0, 1].legend()
            
        # Communication density
        axes[1, 0].plot(metrics['communication_density'])
        axes[1, 0].set_title('Communication Density')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Density')
        
        # Step duration
        axes[1, 1].plot(metrics['step_durations'])
        axes[1, 1].set_title('Step Duration')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Duration (s)')
        
        plt.tight_layout()
        
    def update_animation(self, frame: int) -> List[plt.Artist]:
        """Update animation frame."""
        if not self.paused:
            self.environment.step()
            
        self.ax.clear()
        self.visualize_terrain()
        self.visualize_resources()
        self.visualize_agents()
        self.visualize_communication_network()
        
        return self.ax.get_children()
        
    def start_animation(self, interval: int = 100) -> None:
        """
        Start animation.
        
        Args:
            interval: Animation interval in milliseconds
        """
        self.setup_plot()
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            interval=interval,
            blit=True
        )
        plt.show()
        
    def toggle_pause(self) -> None:
        """Toggle animation pause state."""
        self.paused = not self.paused
        if self.paused:
            self.animation.event_source.stop()
        else:
            self.animation.event_source.start()
            
    def save_animation(self, filepath: str, fps: int = 10, duration: int = 10) -> None:
        """
        Save animation to file.
        
        Args:
            filepath: Path to save animation
            fps: Frames per second
            duration: Duration in seconds
        """
        frames = fps * duration
        self.setup_plot()
        anim = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            frames=frames,
            interval=1000/fps,
            blit=True
        )
        anim.save(filepath, writer='pillow', fps=fps)
        
    def create_snapshot(self, filepath: str) -> None:
        """
        Create snapshot of current state.
        
        Args:
            filepath: Path to save snapshot
        """
        self.setup_plot()
        self.visualize_terrain()
        self.visualize_resources()
        self.visualize_agents()
        self.visualize_communication_network()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
