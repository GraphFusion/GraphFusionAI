
  
![graph-fusion-logo](https://github.com/user-attachments/assets/de5a4a09-a7e4-4b21-b3ec-01d5a3097ecd)

</p>
<h1 align="center" weight='300'>GraphFusionAI: A toolkit for graph-based AI agents and multi-agent systems</h1>
<div align="center">

  [![GitHub release](https://img.shields.io/badge/Github-Release-blue)](https://github.com/GraphFusion/GraphFusion-NMN/releases)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/GraphFusion/GraphFusion/blob/main/LICENSE)
  [![Join us on Teams](https://img.shields.io/badge/Join-Teams-blue)](https://teams.microsoft.com/)
  [![Discord Server](https://img.shields.io/badge/Discord-Server-blue)](https://discord.gg/zK94WvRjZT)

</div>
<h3 align="center">
   <a href="https://github.com/GraphFusion/graphfusion/blob/main/documentation.md"><b>Docs</b></a> &bull;
   <a href="https://graphfusion.github.io/graphfusion.io/"><b>Website</b></a>
</h3>
<br />

# **GraphFusionAI  – The Graph-Based AI Agent Framework**  

⚠️ **This project is in early development!** Expect bugs, incomplete features, and potential breaking changes. If you're contributing, **please read the codebase carefully** and help us improve it.  

GraphFusionAI is an **open-source AI framework** designed to build **graph-powered multi-agent systems** with **neural memory**, **dynamic knowledge graphs**, and **LLM integration**. It enables **autonomous, context-aware AI agents** that can learn, reason, and collaborate in real time.  

## **🌟 Features (Work in Progress)**  
- **Graph-Based Agents** – AI agents that operate on knowledge graphs.  
- **Neural Memory System** – Stores and retrieves contextual knowledge.  
- **Multi-Agent Collaboration** – Agents interact, exchange knowledge, and optimize decisions.  
- **LLM Integration** – Supports OpenAI, Anthropic, LLaMA, and LiteLLM for smart reasoning.  
- **Tool-Based Extensibility** – Agents can use pluggable tools (search, diagnosis, etc.).  
- **Self-Healing Mechanism** – Automatically prunes outdated or irrelevant memory.  
- **Real-Time Learning** – Dynamically updates knowledge and adapts.  

⚠️ **Known Issues:**  
- **Memory module is unstable** – Needs better handling of storage, retrieval, and updates.  
- **Agent decision-making is inconsistent** – Some LLM-based reasoning paths are not fully optimized.  
- **LLM response parsing is unreliable** – Some responses need better structuring.  
- **Graph updates may cause inconsistencies** – Edge cases in knowledge storage require fixes.  
- **Tool integration is not fully tested** – Some tools might not work as expected.  

## **🚀 Quick Start: Create an AI Agent**  

1️⃣ **Install Dependencies**  
```sh
pip install -r requirements.txt
```

2️⃣ **Initialize Graph & Memory Systems**  
```python
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph
from graphfusionai.memory.memory_manager import MemoryManager

graph_network = GraphNetwork()
knowledge_graph = KnowledgeGraph()
memory_manager = MemoryManager()
```

3️⃣ **Build an Agent with LLM & Tools**  
```python
from graphfusionai.agents.agent_builder import AgentBuilder
from graphfusionai.agents.base_agent import BaseAgent

agent_config = {
    "memory": {"input_dim": 256, "memory_dim": 512, "context_dim": 128},
    "llm": {"provider": "openai", "model": "gpt-4", "api_key": "your-api-key"},
    "tools": ["search_tool"]
}

agent_builder = AgentBuilder(graph_network, knowledge_graph)
agent = agent_builder.create_agent(BaseAgent, "AIResearcher", agent_config)
```

4️⃣ **Use Agent for Reasoning & Knowledge Updates**  
```python
query = "What are the latest advancements in AI?"
response = agent.use_llm_for_query(query)
print(response)

agent.update_graph([{"from": "AI", "to": "LLMs", "relation": "advancing"}])
```

## **⚠️ Contributing: Please Read Carefully!**  

This project **is not stable yet!** There are **bugs, missing features, and breaking changes.** If you're contributing, please:  

1️⃣ **Read the codebase carefully** before making any changes.  
2️⃣ **Check the open issues** – See if a bug is already reported before opening a new one.  
3️⃣ **Propose improvements** – If something is broken or inefficient, suggest a better way.  
4️⃣ **Write tests** – Most features are not fully tested yet. Help us improve test coverage.  
5️⃣ **Fix known issues** – Look at the `TODOs` and `FIXME` comments in the code.  

To contribute:  
```sh
git clone https://github.com/GraphFusionAI/GraphFusionAI.git
cd GraphFusionAI
git checkout -b feature-your-fix
# Make changes, commit, and push
```
Then open a **Pull Request (PR)** with a detailed description.  

## **🔧 How You Can Help**
- **Fix Memory System** – Improve memory retrieval, update logic, and self-healing.  
- **Improve Agent Decision-Making** – Optimize reasoning steps and decision-tree logic.  
- **Refactor LLM Response Handling** – Some responses are not structured properly.  
- **Test Tool Integrations** – Make sure tools work correctly and return expected results.  
- **Optimize Knowledge Graph Queries** – Improve search efficiency and update mechanisms.  

## **📌 Roadmap**  
- **LLM Model Agnostic** (OpenAI, Anthropic, LLaMA, etc.)  
- **Graph-Based Decision Making**  
- **Self-Healing Memory**  
- **Multi-Agent Communication**  
- **AutoML for Optimized Agent Performance**  
- **Integration with Hugging Face & LangChain**  

## **💬 Join the Discussion**
Join the conversation, share feedback, and collaborate on new ideas:

- **Discord**: [Join our community](https://discord.gg/zK94WvRjZT)
- **GitHub Discussions**: [Start a thread](https://github.com/GraphFusion/GraphFusionAI/discussions)

🚀 **Let's build the future of graph-based AI together!**

## License

GraphFusionAI is open-source and licensed under the [MIT License](LICENSE).

