"""
Routine discovery SDK wrapper.
"""

from pathlib import Path
from typing import Optional
import os
from openai import OpenAI

from ..routine_discovery.agent import RoutineDiscoveryAgent
from ..routine_discovery.context_manager import ContextManager
from ..data_models.production_routine import Routine


class RoutineDiscovery:
    """
    High-level interface for discovering routines.
    
    Example:
        >>> discovery = RoutineDiscovery(
        ...     client=openai_client,
        ...     task="Search for flights",
        ...     cdp_captures_dir="./captures"
        ... )
        >>> routine = discovery.run()
    """
    
    def __init__(
        self,
        client: OpenAI,
        task: str,
        cdp_captures_dir: str = "./cdp_captures",
        output_dir: str = "./routine_discovery_output",
        llm_model: str = "gpt-5",
    ):
        self.client = client
        self.task = task
        self.cdp_captures_dir = cdp_captures_dir
        self.output_dir = output_dir
        self.llm_model = llm_model
        
        self.agent: Optional[RoutineDiscoveryAgent] = None
        self.context_manager: Optional[ContextManager] = None
    
    def run(self) -> Routine:
        """
        Run routine discovery and return the discovered routine.
        
        Returns:
            Discovered Routine object.
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize context manager
        self.context_manager = ContextManager(
            client=self.client,
            tmp_dir=str(Path(self.output_dir) / "tmp"),
            transactions_dir=str(Path(self.cdp_captures_dir) / "network" / "transactions"),
            consolidated_transactions_path=str(Path(self.cdp_captures_dir) / "network" / "consolidated_transactions.json"),
            storage_jsonl_path=str(Path(self.cdp_captures_dir) / "storage" / "events.jsonl"),
        )
        self.context_manager.make_vectorstore()
        
        # Initialize and run agent
        self.agent = RoutineDiscoveryAgent(
            client=self.client,
            context_manager=self.context_manager,
            task=self.task,
            llm_model=self.llm_model,
            output_dir=self.output_dir,
        )
        self.agent.run()
        
        # Load and return the discovered routine
        routine_path = Path(self.output_dir) / "routine.json"
        if not routine_path.exists():
            raise FileNotFoundError(f"Routine not found at {routine_path}")
        
        return Routine.model_validate_json(routine_path.read_text())

