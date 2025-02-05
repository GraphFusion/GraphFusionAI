# **GraphFusionAI Documentation**

## **Overview**
GraphFusionAI is a powerful toolkit designed for building intelligent AI agents. By leveraging graph-based knowledge systems and neural memory networks, GraphFusionAI empowers developers to create dynamic, adaptable, and collaborative agents. 

The toolkit is ideal for industries like healthcare, education, and customer support, focusing on simplifying agent development with persistent memory, shared knowledge, and real-time adaptability.

## **Features**
- **Agent Builder**: Easily configure and build agents with minimal code.
- **Dynamic Knowledge Graph**: Enable agents to share and update information in real time.
- **Neural Memory Network**: Store, retrieve, and adapt knowledge across agent interactions.
- **Tool Integration**: Dynamically attach tools like email, search, or custom utilities to agents.
- **LLM Integration**: Utilize advanced Large Language Models (LLMs) for natural language understanding and contextual recommendations.

## **Installation**

To install GraphFusionAI, use the following pip command:

```bash
pip install graphfusionai
```
## **Quick Start**
Here’s a step-by-step guide to getting started with GraphFusionAI.

### 1. **Initialize Core Components**
```python
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph

graph_network = GraphNetwork(feature_dim=128, hidden_dim=256)
knowledge_graph = KnowledgeGraph()
```

### 2. **Create an Agent Builder**
```python
from graphfusionai.builder.agent_builder import AgentBuilder

builder = AgentBuilder(graph_network, knowledge_graph)
```

### 3. **Build an Agent**
```python
from graphfusionai.agents.base_agent import BaseAgent

config = {
    "memory": {"input_dim": 128, "memory_dim": 256, "context_dim": 64},
    "tools": ["email", "search"]
}

agent = builder.create_agent(BaseAgent, "SupportAgent", config)
```
## **Configuration**
### **Agent Configuration**
- **`name`**: Unique identifier for the agent.
- **`memory`**: Configurations for the agent's memory system.
    - `input_dim`: Input feature dimension.
    - `memory_dim`: Memory size dimension.
    - `context_dim`: Contextual feature dimension.
- **`tools`**: List of tools to attach to the agent.

### Example:
```json
{
  "memory": {"input_dim": 128, "memory_dim": 256, "context_dim": 64},
  "tools": ["email", "search", "analytics"]
}
```
## **Examples**
### Example 1: Basic Support Agent
```python
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph
from graphfusionai.builder.agent_builder import AgentBuilder
from graphfusionai.agents.base_agent import BaseAgent

graph_network = GraphNetwork(128, 256)
knowledge_graph = KnowledgeGraph()

builder = AgentBuilder(graph_network, knowledge_graph)

config = {
    "memory": {"input_dim": 128, "memory_dim": 256, "context_dim": 64},
    "tools": ["email", "search"]
}

agent = builder.create_agent(BaseAgent, "SupportAgent", config)
agent.learn("Company knowledge base.pdf")
agent.use_tool("email", recipient="user@example.com", message="Hello!")
```

## **Development**
GraphFusionAI welcomes contributions! 

### **Getting Started**
1. Fork the repository.
2. Clone it:
   ```bash
   git clone https://github.com/your-username/GraphFusionAI.git
   cd GraphFusionAI
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run Tests:
   ```bash
   pytest
   ```

## **Contact**
For questions, suggestions, or to report issues:
- **GitHub Issues**: [GraphFusionAI Issues](https://github.com/your-repo/issues)
- **Email**: hello@GraphFusion.onmicrosoft.com
