"""
LLM-specific task types and handlers.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from .task import Task, TaskType
from ..llm import LLMProvider, OpenAIProvider, AnthropicProvider, LLMResponse

class LLMTaskType(Enum):
    """Types of LLM-specific tasks."""
    COMPLETION = "completion"
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"

class LLMTask(Task):
    """
    Task specifically designed for LLM operations.
    
    Features:
    - Multiple LLM provider support
    - Context window management
    - Token optimization
    - Response streaming
    - Prompt templating
    - Chain of thought
    - Few-shot learning
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_type: LLMTaskType,
        prompt: str,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        chain_of_thought: bool = False,
        stream: bool = False,
        **kwargs
    ):
        """Initialize LLM task."""
        super().__init__(
            name=name,
            description=description,
            task_type=TaskType.CUSTOM,
            steps=[{
                "type": "llm",
                "llm_type": llm_type.value,
                "prompt": prompt,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop_sequences": stop_sequences,
                "few_shot_examples": few_shot_examples,
                "chain_of_thought": chain_of_thought,
                "stream": stream
            }],
            **kwargs
        )
        self.llm_type = llm_type
        self.prompt = prompt
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.few_shot_examples = few_shot_examples
        self.chain_of_thought = chain_of_thought
        self.stream = stream
        
        # LLM-specific metrics
        self.metrics.update({
            "token_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0
        })

class LLMTaskExecutor:
    """
    Specialized executor for LLM tasks.
    
    Features:
    - Provider management
    - Context optimization
    - Response validation
    - Error handling
    - Cost tracking
    """
    
    def __init__(self):
        """Initialize LLM task executor."""
        self.providers: Dict[str, LLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider()
        }
    
    def execute(
        self,
        task: LLMTask,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Execute LLM task.
        
        Args:
            task: LLM task to execute
            context: Additional context
            
        Returns:
            LLM response
        """
        # Get provider
        provider = self.providers.get(task.provider)
        if not provider:
            raise ValueError(f"Unknown LLM provider: {task.provider}")
        
        # Build prompt
        prompt = self._build_prompt(task, context)
        
        try:
            # Execute LLM call
            response = provider.generate(
                prompt=prompt,
                model=task.model,
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                stop_sequences=task.stop_sequences,
                stream=task.stream
            )
            
            # Update metrics
            self._update_metrics(task, response)
            
            return response
            
        except Exception as e:
            task.fail(str(e))
            raise
    
    def _build_prompt(
        self,
        task: LLMTask,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build full prompt with context and examples."""
        prompt_parts = []
        
        # Add few-shot examples
        if task.few_shot_examples:
            for example in task.few_shot_examples:
                prompt_parts.append(f"Input: {example['input']}")
                prompt_parts.append(f"Output: {example['output']}\n")
        
        # Add context
        if context:
            prompt_parts.append("Context:")
            for key, value in context.items():
                prompt_parts.append(f"{key}: {value}")
            prompt_parts.append("")
        
        # Add main prompt
        if task.chain_of_thought:
            prompt_parts.append("Let's solve this step by step:")
        prompt_parts.append(task.prompt)
        
        return "\n".join(prompt_parts)
    
    def _update_metrics(
        self,
        task: LLMTask,
        response: LLMResponse
    ) -> None:
        """Update task metrics with response data."""
        task.metrics["token_count"] = response.token_count
        task.metrics["prompt_tokens"] = response.prompt_tokens
        task.metrics["completion_tokens"] = response.completion_tokens
        task.metrics["total_cost"] = response.total_cost

def create_completion_task(
    prompt: str,
    name: Optional[str] = None,
    **kwargs
) -> LLMTask:
    """Create a simple completion task."""
    return LLMTask(
        name=name or "Completion Task",
        description="Generate completion for prompt",
        llm_type=LLMTaskType.COMPLETION,
        prompt=prompt,
        **kwargs
    )

def create_chat_task(
    messages: List[Dict[str, str]],
    name: Optional[str] = None,
    **kwargs
) -> LLMTask:
    """Create a chat task."""
    prompt = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in messages
    )
    return LLMTask(
        name=name or "Chat Task",
        description="Chat conversation",
        llm_type=LLMTaskType.CHAT,
        prompt=prompt,
        **kwargs
    )

def create_code_task(
    prompt: str,
    task_type: LLMTaskType,
    name: Optional[str] = None,
    **kwargs
) -> LLMTask:
    """Create a code-related task."""
    if task_type not in [
        LLMTaskType.CODE_GENERATION,
        LLMTaskType.CODE_REVIEW
    ]:
        raise ValueError("Invalid code task type")
    
    return LLMTask(
        name=name or f"Code {task_type.value.title()} Task",
        description=f"Perform code {task_type.value}",
        llm_type=task_type,
        prompt=prompt,
        temperature=0.2,  # Lower temperature for code tasks
        **kwargs
    )

def create_analysis_task(
    data: Any,
    analysis_type: str,
    name: Optional[str] = None,
    **kwargs
) -> LLMTask:
    """Create an analysis task."""
    prompt = f"""
    Analyze the following data:
    {data}
    
    Analysis type: {analysis_type}
    
    Please provide:
    1. Key findings
    2. Supporting evidence
    3. Recommendations
    """
    return LLMTask(
        name=name or "Analysis Task",
        description=f"Analyze data using {analysis_type}",
        llm_type=LLMTaskType.ANALYSIS,
        prompt=prompt,
        chain_of_thought=True,  # Enable step-by-step analysis
        **kwargs
    )
