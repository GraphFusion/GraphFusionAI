"""
Task execution and monitoring.
"""
from typing import Dict, Any
from datetime import datetime
from .task import Task, TaskStatus
from ..agents import BaseAgent
from ..memory import MemoryManager
from ..knowledge_graph import KnowledgeGraph

class TaskExecutor:
    """
    Executes tasks using assigned agents.
    
    Features:
    - Step-by-step execution
    - Progress monitoring
    - Result validation
    - Error handling
    - Resource management
    """
    
    def execute_task(
        self,
        task: Task,
        agent: BaseAgent,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """
        Execute task using assigned agent.
        
        Args:
            task: Task to execute
            agent: Agent to use
            memory: Memory manager
            knowledge_graph: Knowledge graph
            
        Returns:
            Execution result
        """
        result = {
            "status": TaskStatus.PENDING,
            "steps_completed": 0,
            "output": [],
            "error": None,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "metrics": {}
        }
        
        try:
            # Start task
            task.start()
            result["status"] = TaskStatus.IN_PROGRESS
            
            # Execute each step
            for step in task.steps:
                step_result = self._execute_step(
                    step,
                    agent,
                    memory,
                    knowledge_graph
                )
                
                result["output"].append(step_result)
                result["steps_completed"] += 1
                
                # Check step result
                if not step_result.get("success", False):
                    raise Exception(
                        f"Step {result['steps_completed']} failed: "
                        f"{step_result.get('error')}"
                    )
            
            # Task completed successfully
            result["status"] = TaskStatus.COMPLETED
            
        except Exception as e:
            # Task failed
            result["status"] = TaskStatus.FAILED
            result["error"] = str(e)
        
        # Record end time and duration
        result["end_time"] = datetime.now().isoformat()
        result["metrics"]["duration"] = (
            datetime.fromisoformat(result["end_time"]) -
            datetime.fromisoformat(result["start_time"])
        ).total_seconds()
        
        return result
    
    def _execute_step(
        self,
        step: Dict[str, Any],
        agent: BaseAgent,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Execute single task step."""
        step_result = {
            "step_type": step["type"],
            "success": False,
            "output": None,
            "error": None,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Get step handler based on type
            handler = getattr(
                self,
                f"_handle_{step['type']}_step",
                self._handle_default_step
            )
            
            # Execute step
            output = handler(
                step,
                agent,
                memory,
                knowledge_graph
            )
            
            step_result["success"] = True
            step_result["output"] = output
            
        except Exception as e:
            step_result["error"] = str(e)
        
        # Record end time
        step_result["end_time"] = datetime.now().isoformat()
        step_result["duration"] = (
            datetime.fromisoformat(step_result["end_time"]) -
            datetime.fromisoformat(step_result["start_time"])
        ).total_seconds()
        
        return step_result
    
    def _handle_default_step(
        self,
        step: Dict[str, Any],
        agent: BaseAgent,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Any:
        """Default step handler."""
        # Get step input
        step_input = self._get_step_input(
            step,
            memory,
            knowledge_graph
        )
        
        # Execute step using agent
        return agent.execute_task(step_input)
    
    def _handle_research_step(
        self,
        step: Dict[str, Any],
        agent: BaseAgent,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Any:
        """Handle research step."""
        query = step["query"]
        
        # Check memory first
        memory_results = memory.search_memories(query)
        if memory_results:
            return {
                "source": "memory",
                "results": memory_results
            }
        
        # Check knowledge graph
        graph_results = knowledge_graph.search(query)
        if graph_results:
            return {
                "source": "knowledge_graph",
                "results": graph_results
            }
        
        # Perform new research
        research_results = agent.research(query)
        
        # Store results
        memory.add_memory({
            "type": "research",
            "query": query,
            "results": research_results,
            "timestamp": datetime.now().isoformat()
        })
        
        knowledge_graph.add_research_results(
            query,
            research_results
        )
        
        return {
            "source": "new_research",
            "results": research_results
        }
    
    def _handle_analysis_step(
        self,
        step: Dict[str, Any],
        agent: BaseAgent,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Any:
        """Handle analysis step."""
        data = step["data"]
        analysis_type = step["analysis_type"]
        
        # Get relevant context
        context = self._get_analysis_context(
            analysis_type,
            memory,
            knowledge_graph
        )
        
        # Perform analysis
        analysis_results = agent.analyze(
            data,
            analysis_type,
            context
        )
        
        # Store results
        knowledge_graph.add_analysis_results(
            data,
            analysis_type,
            analysis_results
        )
        
        return analysis_results
    
    def _handle_decision_step(
        self,
        step: Dict[str, Any],
        agent: BaseAgent,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Any:
        """Handle decision step."""
        options = step["options"]
        criteria = step["criteria"]
        
        # Get decision context
        context = self._get_decision_context(
            criteria,
            memory,
            knowledge_graph
        )
        
        # Make decision
        decision = agent.make_decision(
            options,
            criteria,
            context
        )
        
        # Store decision
        memory.add_memory({
            "type": "decision",
            "options": options,
            "criteria": criteria,
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        })
        
        return decision
    
    def _handle_llm_step(
        self,
        step: Dict[str, Any],
        agent: BaseAgent,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Any:
        """Handle LLM step."""
        from ..llm import get_llm_provider
        
        # Get LLM provider
        provider = get_llm_provider(step.get("provider", "default"))
        
        # Get context
        context = self._get_llm_context(
            step,
            memory,
            knowledge_graph
        )
        
        # Add context to prompt if needed
        prompt = step["prompt"]
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
        
        # Execute LLM call
        response = provider.generate(
            prompt=prompt,
            **{k: v for k, v in step.items() 
               if k in ["model", "temperature", "max_tokens", "stop"]}
        )
        
        # Store response in memory
        memory.add_memory({
            "type": "llm_response",
            "prompt": prompt,
            "response": response.text,
            "metrics": response.metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update knowledge graph
        knowledge_graph.add_node(
            f"llm_response_{datetime.now().isoformat()}",
            type="llm_response",
            prompt=prompt,
            response=response.text,
            metrics=response.metrics
        )
        
        return response.text
    
    def _get_step_input(
        self,
        step: Dict[str, Any],
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Get input for step execution."""
        step_input = step.copy()
        
        # Add context from memory
        if "context_query" in step:
            memories = memory.search_memories(
                step["context_query"]
            )
            step_input["context"] = memories
        
        # Add data from knowledge graph
        if "knowledge_query" in step:
            knowledge = knowledge_graph.search(
                step["knowledge_query"]
            )
            step_input["knowledge"] = knowledge
        
        return step_input
    
    def _get_analysis_context(
        self,
        analysis_type: str,
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Get context for analysis."""
        context = {}
        
        # Get relevant memories
        memories = memory.search_memories(
            f"analysis {analysis_type}"
        )
        if memories:
            context["previous_analyses"] = memories
        
        # Get relevant knowledge
        knowledge = knowledge_graph.get_analysis_patterns(
            analysis_type
        )
        if knowledge:
            context["patterns"] = knowledge
        
        return context
    
    def _get_decision_context(
        self,
        criteria: Dict[str, Any],
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Get context for decision making."""
        context = {}
        
        # Get relevant memories
        for criterion in criteria:
            memories = memory.search_memories(criterion)
            if memories:
                context[f"{criterion}_history"] = memories
        
        # Get decision patterns
        patterns = knowledge_graph.get_decision_patterns()
        if patterns:
            context["patterns"] = patterns
        
        return context
    
    def _get_llm_context(
        self,
        step: Dict[str, Any],
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Get context for LLM task."""
        context = {}
        
        # Get relevant memories
        memories = memory.search_memories(
            f"llm {step['llm_type']}"
        )
        if memories:
            context["previous_responses"] = memories
        
        # Get relevant knowledge
        knowledge = knowledge_graph.get_llm_patterns(
            step["llm_type"]
        )
        if knowledge:
            context["patterns"] = knowledge
        
        return context
