"""
Routine discovery SDK wrapper.
"""

from pathlib import Path
from typing import Optional, Callable
import os
import json
from openai import OpenAI
from pydantic import BaseModel

from ..routine_discovery.agent import RoutineDiscoveryAgent
from ..routine_discovery.context_manager import LocalContextManager
from ..data_models.routine.routine import Routine
from ..data_models.routine_discovery.message import RoutineDiscoveryMessage
from ..data_models.routine_discovery.llm_responses import TestParametersResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RoutineDiscoveryResult(BaseModel):
    """Result of routine discovery containing the routine and test parameters."""
    routine: Routine
    test_parameters: TestParametersResponse


class RoutineDiscovery:
    """
    High-level interface for discovering routines.

    Example:
        >>> discovery = RoutineDiscovery(
        ...     client=openai_client,
        ...     task="Search for flights",
        ...     cdp_captures_dir="./captures"
        ... )
        >>> result = discovery.run()
        >>> routine = result.routine
        >>> test_params = result.test_parameters
    """

    def __init__(
        self,
        client: OpenAI,
        task: str,
        cdp_captures_dir: str = "./cdp_captures",
        output_dir: str = "./routine_discovery_output",
        llm_model: str = "gpt-5.1",
        message_callback: Optional[Callable[[RoutineDiscoveryMessage], None]] = None,
    ):
        """
        Initialize the RoutineDiscovery SDK.

        Args:
            client: OpenAI client instance
            task: Description of the task to discover routines for
            cdp_captures_dir: Directory containing CDP captures
            output_dir: Directory to save output files
            llm_model: LLM model to use for discovery
            message_callback: Optional callback for progress messages
        """
        self.client = client
        self.task = task
        self.cdp_captures_dir = cdp_captures_dir
        self.output_dir = output_dir
        self.llm_model = llm_model
        self.message_callback = message_callback or self._default_message_handler

        self.agent: Optional[RoutineDiscoveryAgent] = None
        self.context_manager: Optional[LocalContextManager] = None

    def _default_message_handler(self, message: RoutineDiscoveryMessage) -> None:
        """Default message handler that logs to console."""
        from ..data_models.routine_discovery.message import RoutineDiscoveryMessageType

        if message.type == RoutineDiscoveryMessageType.INITIATED:
            logger.info(f"🚀 {message.content}")
        elif message.type == RoutineDiscoveryMessageType.PROGRESS_THINKING:
            logger.info(f"🤔 {message.content}")
        elif message.type == RoutineDiscoveryMessageType.PROGRESS_RESULT:
            logger.info(f"✅ {message.content}")
        elif message.type == RoutineDiscoveryMessageType.FINISHED:
            logger.info(f"🎉 {message.content}")
        elif message.type == RoutineDiscoveryMessageType.ERROR:
            logger.error(f"❌ {message.content}")
    
    def run(self) -> RoutineDiscoveryResult:
        """
        Run routine discovery and return the discovered routine with test parameters.

        Returns:
            RoutineDiscoveryResult containing the routine and test parameters.
        """
        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Initialize context manager
            self.context_manager = LocalContextManager(
                client=self.client,
                tmp_dir=str(Path(self.output_dir) / "tmp"),
                transactions_dir=str(Path(self.cdp_captures_dir) / "network" / "transactions"),
                consolidated_transactions_path=str(Path(self.cdp_captures_dir) / "network" / "consolidated_transactions.json"),
                storage_jsonl_path=str(Path(self.cdp_captures_dir) / "storage" / "events.jsonl"),
                window_properties_path=str(Path(self.cdp_captures_dir) / "window_properties" / "window_properties.json"),
            )
            logger.info("Context manager initialized.")

            # Make the vectorstore
            self.context_manager.make_vectorstore()
            logger.info(f"Vectorstore created: {self.context_manager.vectorstore_id}")

            # Initialize and run agent
            self.agent = RoutineDiscoveryAgent(
                client=self.client,
                context_manager=self.context_manager,
                task=self.task,
                emit_message_callable=self.message_callback,
                llm_model=self.llm_model,
                output_dir=self.output_dir,
            )

            # Run agent and get routine
            routine = self.agent.run()
            logger.info("Routine discovery completed successfully.")

            # Get test parameters from the agent
            test_parameters = self.agent.get_test_parameters(routine)
            logger.info("Test parameters generated successfully.")

            # Save test parameters to output directory
            test_params_path = os.path.join(self.output_dir, "test_parameters.json")
            with open(test_params_path, mode="w", encoding="utf-8") as f:
                json.dump(test_parameters.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"Test parameters saved to: {test_params_path}")

            return RoutineDiscoveryResult(
                routine=routine,
                test_parameters=test_parameters
            )

        finally:
            # Clean up vectorstore
            if self.context_manager is not None and self.context_manager.vectorstore_id is not None:
                logger.info("Cleaning up vectorstore...")
                self.context_manager.clean_up()
                logger.info("Vectorstore cleaned up.")

