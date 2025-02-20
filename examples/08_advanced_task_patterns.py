"""
Advanced example demonstrating complex task patterns in GraphFusionAI.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import random
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class TaskType(Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REVIEW = "review"
    DECISION = "decision"

class BaseAgent:
    def __init__(self, name: str, skills: List[str] = None):
        self.name = name
        self.id = name.lower().replace(" ", "_")
        self.skills = skills or []
        self.role = None
        self.is_busy = False
        self.current_task = None
        self.performance_stats = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "avg_completion_time": 0.0
        }
    
    def execute_step(self, step: dict, context: dict) -> dict:
        # Simulate varying success rates and completion times
        success = random.random() > 0.2  # 80% success rate
        completion_time = random.uniform(0.5, 2.0)  # 0.5-2.0 time units
        
        result = {
            "status": "completed" if success else "failed",
            "completion_time": completion_time,
            "result": f"Executed {step['type']} with {'success' if success else 'failure'}"
        }
        
        # Update performance stats
        self.performance_stats["tasks_completed"] += 1
        self.performance_stats["success_rate"] = (
            (self.performance_stats["success_rate"] * (self.performance_stats["tasks_completed"] - 1) + int(success))
            / self.performance_stats["tasks_completed"]
        )
        self.performance_stats["avg_completion_time"] = (
            (self.performance_stats["avg_completion_time"] * (self.performance_stats["tasks_completed"] - 1) + completion_time)
            / self.performance_stats["tasks_completed"]
        )
        
        return result

class TaskManager:
    """Advanced task manager with complex execution patterns."""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.task_graph: Dict[str, List[str]] = {}  # task_id -> dependent_task_ids
        self.execution_history: List[Dict] = []
        
    def add_task(self, task: Dict) -> None:
        """Add a task with its dependencies."""
        self.tasks[task["id"]] = task
        # Initialize task graph
        self.task_graph[task["id"]] = []
        for dep in task.get("dependencies", []):
            if dep in self.task_graph:
                self.task_graph[dep].append(task["id"])
    
    def get_ready_tasks(self) -> List[Dict]:
        """Get tasks ready for execution (all dependencies met)."""
        ready_tasks = []
        for task_id, task in self.tasks.items():
            if task["status"] == TaskStatus.PENDING.value:
                deps_met = all(
                    self.tasks[dep]["status"] == TaskStatus.COMPLETED.value
                    for dep in task.get("dependencies", [])
                )
                if deps_met:
                    ready_tasks.append(task)
        return ready_tasks
    
    def execute_tasks(self, agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """Execute tasks with complex patterns."""
        results = {
            "completed": [],
            "failed": [],
            "skipped": [],
            "metrics": {
                "total_time": 0,
                "success_rate": 0,
                "parallel_execution": 0
            }
        }
        
        while True:
            ready_tasks = self.get_ready_tasks()
            if not ready_tasks:
                break
            
            # Track parallel execution
            results["metrics"]["parallel_execution"] = max(
                results["metrics"]["parallel_execution"],
                len(ready_tasks)
            )
            
            # Execute ready tasks in parallel
            for task in ready_tasks:
                # Find best agent for task
                best_agent = self._find_best_agent(task, agents)
                if not best_agent:
                    task["status"] = TaskStatus.SKIPPED.value
                    results["skipped"].append(task["id"])
                    continue
                
                # Execute task steps
                task["status"] = TaskStatus.IN_PROGRESS.value
                task_result = self._execute_task_steps(task, best_agent)
                
                # Update task status
                if task_result["status"] == "completed":
                    task["status"] = TaskStatus.COMPLETED.value
                    results["completed"].append(task["id"])
                else:
                    task["status"] = TaskStatus.FAILED.value
                    results["failed"].append(task["id"])
                    # Handle failure based on task type
                    self._handle_task_failure(task, results)
                
                # Record execution
                self.execution_history.append({
                    "task_id": task["id"],
                    "agent_id": best_agent.id,
                    "result": task_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update metrics
                results["metrics"]["total_time"] += task_result.get("completion_time", 0)
        
        # Calculate final metrics
        total_tasks = len(results["completed"]) + len(results["failed"])
        if total_tasks > 0:
            results["metrics"]["success_rate"] = len(results["completed"]) / total_tasks
        
        return results
    
    def _find_best_agent(self, task: Dict, agents: Dict[str, BaseAgent]) -> Optional[BaseAgent]:
        """Find best agent for task based on skills and performance."""
        qualified_agents = []
        required_skills = set(task.get("required_skills", []))
        
        for agent in agents.values():
            if not agent.is_busy and required_skills.issubset(set(agent.skills)):
                # Calculate agent score
                skill_match = len(required_skills & set(agent.skills)) / len(required_skills)
                success_rate = agent.performance_stats["success_rate"]
                speed = 1 / (agent.performance_stats["avg_completion_time"] + 0.1)  # Avoid div by 0
                
                score = (skill_match * 0.4) + (success_rate * 0.4) + (speed * 0.2)
                qualified_agents.append((score, agent))
        
        if qualified_agents:
            return max(qualified_agents, key=lambda x: x[0])[1]  # Sort by score and return best agent
        return None
    
    def _execute_task_steps(self, task: Dict, agent: BaseAgent) -> Dict:
        """Execute task steps with the assigned agent."""
        results = []
        total_time = 0
        
        for step in task["steps"]:
            # Check for conditional execution
            if step.get("condition"):
                if not self._evaluate_condition(step["condition"], results):
                    continue
            
            # Execute step
            result = agent.execute_step(step, {"previous_results": results})
            results.append(result)
            total_time += result.get("completion_time", 0)
            
            # Handle dynamic task generation
            if step.get("generates_tasks"):
                self._generate_dynamic_tasks(step, result)
            
            # Break on failure if specified
            if result["status"] != "completed" and step.get("fail_fast", False):
                return {
                    "status": "failed",
                    "results": results,
                    "completion_time": total_time
                }
        
        # Determine overall status
        success = all(r["status"] == "completed" for r in results)
        return {
            "status": "completed" if success else "failed",
            "results": results,
            "completion_time": total_time
        }
    
    def _evaluate_condition(self, condition: Dict, previous_results: List[Dict]) -> bool:
        """Evaluate condition for conditional execution."""
        condition_type = condition.get("type")
        if condition_type == "all_success":
            return all(r["status"] == "completed" for r in previous_results)
        elif condition_type == "any_success":
            return any(r["status"] == "completed" for r in previous_results)
        elif condition_type == "threshold":
            success_rate = sum(1 for r in previous_results if r["status"] == "completed")
            return success_rate >= condition["value"]
        return True
    
    def _generate_dynamic_tasks(self, step: Dict, result: Dict) -> None:
        """Generate dynamic tasks based on step results."""
        if result["status"] == "completed":
            new_tasks = step["generates_tasks"](result)
            for task in new_tasks:
                self.add_task(task)
    
    def _handle_task_failure(self, task: Dict, results: Dict) -> None:
        """Handle task failure based on task type and configuration."""
        failure_strategy = task.get("failure_strategy", "skip_dependent")
        
        if failure_strategy == "skip_dependent":
            # Mark all dependent tasks as skipped
            for dep_id in self.task_graph[task["id"]]:
                self.tasks[dep_id]["status"] = TaskStatus.SKIPPED.value
                results["skipped"].append(dep_id)
        
        elif failure_strategy == "retry":
            # Add retry task
            retry_task = {
                **task,
                "id": f"{task['id']}_retry",
                "name": f"Retry: {task['name']}",
                "priority": task["priority"] + 1,
                "dependencies": task.get("dependencies", [])[:]  # Copy original dependencies
            }
            self.add_task(retry_task)
        
        elif failure_strategy == "alternate":
            # Add alternate task if specified
            if "alternate_task" in task:
                self.add_task(task["alternate_task"])

def create_complex_task_workflow() -> List[Dict]:
    """Create a complex task workflow with various patterns."""
    tasks = []
    
    # 1. Initial Research Tasks (Parallel)
    for topic in ["quantum_hardware", "quantum_algorithms", "error_correction"]:
        research_task = {
            "id": f"research_{topic}",
            "name": f"Research {topic.replace('_', ' ').title()}",
            "type": TaskType.RESEARCH.value,
            "description": f"Research latest developments in {topic}",
            "steps": [
                {
                    "type": "research",
                    "topic": topic,
                    "fail_fast": True  # Stop on first failure
                },
                {
                    "type": "analyze",
                    "focus": "key findings",
                    "condition": {
                        "type": "all_success",
                        "previous_steps": ["research"]
                    }
                }
            ],
            "status": TaskStatus.PENDING.value,
            "priority": 5,
            "required_skills": ["research", "analysis"],
            "dependencies": [],
            "failure_strategy": "retry"
        }
        tasks.append(research_task)
    
    # 2. Analysis Task (Depends on all research)
    analysis_task = {
        "id": "comprehensive_analysis",
        "name": "Comprehensive Analysis",
        "type": TaskType.ANALYSIS.value,
        "description": "Analyze findings from all research tasks",
        "steps": [
            {
                "type": "analyze",
                "focus": "cross-topic implications",
                "generates_tasks": lambda result: [
                    {
                        "id": "deep_dive",
                        "name": "Deep Dive Analysis",
                        "type": TaskType.ANALYSIS.value,
                        "steps": [{"type": "analyze", "focus": "specific area"}],
                        "status": TaskStatus.PENDING.value,
                        "priority": 4,
                        "required_skills": ["analysis"],
                        "dependencies": ["comprehensive_analysis"]
                    }
                ] if result["status"] == "completed" else []
            }
        ],
        "status": TaskStatus.PENDING.value,
        "priority": 4,
        "required_skills": ["analysis", "synthesis"],
        "dependencies": [
            "research_quantum_hardware",
            "research_quantum_algorithms",
            "research_error_correction"
        ],
        "failure_strategy": "alternate",
        "alternate_task": {
            "id": "simplified_analysis",
            "name": "Simplified Analysis",
            "type": TaskType.ANALYSIS.value,
            "steps": [{"type": "analyze", "focus": "basic findings"}],
            "status": TaskStatus.PENDING.value,
            "priority": 4,
            "required_skills": ["analysis"],
            "dependencies": [
                "research_quantum_hardware",
                "research_quantum_algorithms",
                "research_error_correction"
            ]
        }
    }
    tasks.append(analysis_task)
    
    # 3. Design Tasks (Conditional Branch)
    design_task = {
        "id": "system_design",
        "name": "System Design",
        "type": TaskType.DESIGN.value,
        "description": "Design quantum computing system",
        "steps": [
            {
                "type": "design",
                "focus": "architecture",
                "condition": {
                    "type": "threshold",
                    "value": 0.7  # 70% success rate required
                }
            }
        ],
        "status": TaskStatus.PENDING.value,
        "priority": 3,
        "required_skills": ["system_design", "quantum_computing"],
        "dependencies": ["comprehensive_analysis"],
        "failure_strategy": "skip_dependent"
    }
    tasks.append(design_task)
    
    # 4. Implementation Tasks (Parallel with Dependencies)
    for component in ["core", "interface", "optimizer"]:
        implement_task = {
            "id": f"implement_{component}",
            "name": f"Implement {component.title()}",
            "type": TaskType.IMPLEMENTATION.value,
            "description": f"Implement {component} component",
            "steps": [
                {
                    "type": "implement",
                    "component": component
                },
                {
                    "type": "test",
                    "focus": "unit tests"
                }
            ],
            "status": TaskStatus.PENDING.value,
            "priority": 2,
            "required_skills": ["coding", "testing"],
            "dependencies": ["system_design"]
        }
        tasks.append(implement_task)
    
    # 5. Integration Task (Final)
    integration_task = {
        "id": "integration",
        "name": "System Integration",
        "type": TaskType.IMPLEMENTATION.value,
        "description": "Integrate all components",
        "steps": [
            {
                "type": "implement",
                "focus": "integration"
            },
            {
                "type": "test",
                "focus": "integration tests"
            }
        ],
        "status": TaskStatus.PENDING.value,
        "priority": 1,
        "required_skills": ["integration", "testing"],
        "dependencies": [
            "implement_core",
            "implement_interface",
            "implement_optimizer"
        ]
    }
    tasks.append(integration_task)
    
    return tasks

def create_specialized_agents() -> Dict[str, BaseAgent]:
    """Create specialized agents for the workflow."""
    agents = {}
    
    # Research Team
    agents["senior_researcher"] = BaseAgent(
        "Senior Researcher",
        skills=["research", "analysis", "synthesis", "quantum_physics"]
    )
    agents["research_assistant"] = BaseAgent(
        "Research Assistant",
        skills=["research", "documentation", "analysis"]
    )
    
    # Technical Team
    agents["architect"] = BaseAgent(
        "System Architect",
        skills=["system_design", "quantum_computing", "architecture"]
    )
    agents["senior_dev"] = BaseAgent(
        "Senior Developer",
        skills=["coding", "system_design", "integration", "testing"]
    )
    agents["developer"] = BaseAgent(
        "Developer",
        skills=["coding", "testing", "documentation"]
    )
    
    # QA Team
    agents["qa_lead"] = BaseAgent(
        "QA Lead",
        skills=["testing", "quality_assurance", "integration"]
    )
    
    return agents

def main():
    """Run the advanced task patterns example."""
    print("🚀 Advanced Task Patterns Demo")
    print("=============================")
    
    try:
        # Create task manager
        task_manager = TaskManager()
        
        # Create workflow
        print("\n1. Creating Complex Task Workflow...")
        tasks = create_complex_task_workflow()
        for task in tasks:
            print(f"\nTask: {task['name']}")
            print(f"Type: {task['type']}")
            print(f"Dependencies: {len(task.get('dependencies', []))}")
            task_manager.add_task(task)
        
        # Create agents
        print("\n2. Creating Specialized Agents...")
        agents = create_specialized_agents()
        for agent_id, agent in agents.items():
            print(f"\nAgent: {agent.name}")
            print(f"Skills: {', '.join(agent.skills)}")
        
        # Execute workflow
        print("\n3. Executing Task Workflow...")
        results = task_manager.execute_tasks(agents)
        
        # Show results
        print("\n4. Execution Results:")
        print(f"Completed Tasks: {len(results['completed'])}")
        print(f"Failed Tasks: {len(results['failed'])}")
        print(f"Skipped Tasks: {len(results['skipped'])}")
        print("\nMetrics:")
        print(f"Total Time: {results['metrics']['total_time']:.2f} units")
        print(f"Success Rate: {results['metrics']['success_rate']:.2%}")
        print(f"Max Parallel Tasks: {results['metrics']['parallel_execution']}")
        
        # Show execution history
        print("\n5. Execution History:")
        for entry in task_manager.execution_history[-5:]:  # Show last 5 entries
            print(f"\nTask: {entry['task_id']}")
            print(f"Agent: {entry['agent_id']}")
            print(f"Status: {entry['result']['status']}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
