# GraphFusionAI Examples

This directory contains example scripts demonstrating various features of GraphFusionAI.

## Basic Examples

### 1. Basic Agent (`01_basic_agent.py`)
Demonstrates core functionality including:
- Creating and configuring a basic agent
- Using the memory system
- Using the LLM client
- Using tools (web search)

```bash
python examples/01_basic_agent.py
```

### 2. Multi-Agent Collaboration (`02_multi_agent_collaboration.py`)
Shows how multiple specialized agents can work together:
- Research Agent: Gathers information
- Analysis Agent: Analyzes findings
- Summary Agent: Creates concise summaries
- Shared memory and knowledge

```bash
python examples/02_multi_agent_collaboration.py
```

### 3. Knowledge Graph Reasoning (`03_knowledge_graph_reasoning.py`)
Demonstrates the knowledge graph capabilities:
- Building knowledge graphs from research
- Querying the knowledge graph
- Explaining relationships between entities
- Knowledge-based reasoning

```bash
python examples/03_knowledge_graph_reasoning.py
```

### 4. Simulation System (`04_simulation_agents.py`)
Shows the simulation system in action:
- Complex environment with various terrain types
- Multiple resource types with regeneration
- Specialized agents (Explorer, Gatherer, etc.)
- Real-time visualization
- Performance metrics

```bash
python examples/04_simulation_agents.py
```

## Prerequisites

Before running the examples:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
- For web search examples: Get a Tavily API key
- For LLM examples: Configure your preferred LLM provider

3. Configure the keys in the example scripts or use environment variables:
```bash
export TAVILY_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

## Notes

- These examples are designed to demonstrate various features of GraphFusionAI
- Each example includes detailed comments explaining the code
- The simulation example includes visualization that requires a display
- Some examples may take time to run due to API calls and processing

## Troubleshooting

Common issues:
1. **API Key Errors**: Make sure you've set up all required API keys
2. **Import Errors**: Ensure you've installed all dependencies
3. **Display Issues**: For simulation, ensure you have a working display
4. **Memory Issues**: Large knowledge graphs may require significant RAM
