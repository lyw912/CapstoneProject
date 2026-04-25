"""
Report Engine node base class.

All high-level reasoning nodes inherit from this, providing unified logging, input validation, and state mutation interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..llms.base import LLMClient
from ..state.state import ReportState
from loguru import logger

class BaseNode(ABC):
    """
    Node base class.

    Provides unified logging utilities, input/output hooks, and LLM client dependency injection,
    allowing all nodes to focus solely on business logic.
    """
    
    def __init__(self, llm_client: LLMClient, node_name: str = ""):
        """
        Initialize the node.
        
        Args:
            llm_client: LLM client instance
            node_name: Name of the node

        BaseNode saves the node name for unified log prefix output.
        """
        self.llm_client = llm_client
        self.node_name = node_name or self.__class__.__name__
    
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """
        Execute node processing logic.
        
        Args:
            input_data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Processing result
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.
        Defaults to passing directly; subclasses can override for field checking.
        
        Args:
            input_data: Input data
            
        Returns:
            Whether validation passed
        """
        return True
    
    def process_output(self, output: Any) -> Any:
        """
        Process output data.
        Subclasses can override for structuring or validation.
        
        Args:
            output: Raw output
            
        Returns:
            Processed output
        """
        return output
    
    def log_info(self, message: str):
        """Log info message with node name as prefix automatically."""
        formatted_message = f"[{self.node_name}] {message}"
        logger.info(formatted_message)
    
    def log_error(self, message: str):
        """Log error message for troubleshooting."""
        formatted_message = f"[{self.node_name}] {message}"
        logger.error(formatted_message)


class StateMutationNode(BaseNode):
    """
    Node base class with state mutation capability.

    Suitable for scenarios where nodes need to write directly to ReportState.
    """
    
    @abstractmethod
    def mutate_state(self, input_data: Any, state: ReportState, **kwargs) -> ReportState:
        """
        Mutate state.

        Subclasses must return a new state object or pass back after in-place modification for pipeline recording.
        
        Args:
            input_data: Input data
            state: Current state
            **kwargs: Additional parameters
            
        Returns:
            Modified state
        """
        pass
