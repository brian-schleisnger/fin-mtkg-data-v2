from dataclasses import dataclass, field
from agent.memory import DataFrameMemory, ContextOptimizer

@dataclass
class SessionContext:
    """
    Holds all stateful information and memory objects required by the agent during a run.
    By passing this context object into the agent loop, we decouple the core logic 
    from UI frameworks like Streamlit, enabling CLI execution and automated testing.
    """
    
    # Token Tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost Tracking
    estimated_cost: float = 0.0
    
    # Memory and Optimization Instances
    # Using default_factory ensures each SessionContext gets its own isolated instance
    df_memory: DataFrameMemory = field(default_factory=DataFrameMemory)
    context_optimizer: ContextOptimizer = field(default_factory=ContextOptimizer)