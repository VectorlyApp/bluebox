"""
bluebox/sdk/discovery.py

Routine discovery SDK wrapper.

Contains:
- RoutineDiscoveryResult: Discovered routine with messages and vectorstore IDs
- RoutineDiscovery: Wraps RoutineDiscoveryAgent for easy routine generation
- discover(): Generate routine from CDP captures using LLM
- Uses: LocalDiscoveryDataStore, RoutineDiscoveryAgent
"""

import json
import os
from pathlib import Path
from typing import Callable
from openai import OpenAI
from pydantic import BaseModel

# Package root for code_paths (bluebox/sdk/ -> bluebox/)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

from ..agents.routine_discovery_agent import RoutineDiscoveryAgent
from ..llms.infra.data_store import LocalDiscoveryDataStore
from ..llms.llm_client import LLMClient
from ..data_models.llms.vendors import OpenAIModel
from ..data_models.routine.routine import Routine
from ..data_models.routine_discovery.message import RoutineDiscoveryMessage
from ..data_models.routine_discovery.llm_responses import TestParametersResponse
from ..data_models.routine_discovery.message import RoutineDiscoveryMessageType
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
        message_callback: Callable[[RoutineDiscoveryMessage], None] | None = None,
        remote_debugging_address: str | None = None,
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
            remote_debugging_address: Chrome remote debugging address for routine validation.
                If provided, the agent will execute the routine to validate it works.
        """
        self.client = client
        self.task = task
        self.cdp_captures_dir = cdp_captures_dir
        self.output_dir = output_dir
        self.llm_model = llm_model
        self.message_callback = message_callback or self._default_message_handler
        self.remote_debugging_address = remote_debugging_address

        self.agent: RoutineDiscoveryAgent | None = None
        self.data_store: LocalDiscoveryDataStore | None = None

    def _default_message_handler(self, message: RoutineDiscoveryMessage) -> None:
        """Default message handler that logs to console."""

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

            # Initialize data store
            self.data_store = LocalDiscoveryDataStore(
                client=self.client,
                cdp_captures_dir=self.cdp_captures_dir,
                tmp_dir=str(Path(self.output_dir) / "tmp"),
                documentation_paths=[str(PACKAGE_ROOT / "agent_docs")],
                code_paths=[
                    str(PACKAGE_ROOT / "data_models" / "routine"),
                    str(PACKAGE_ROOT / "data_models" / "ui_elements.py"),
                    str(PACKAGE_ROOT / "utils" / "js_utils.py"),
                    str(PACKAGE_ROOT / "utils" / "data_utils.py"),
                    "!" + str(PACKAGE_ROOT / "**" / "__init__.py"),
                ],
            )
            logger.info("Data store initialized.")

            # Make the vectorstores
            self.data_store.make_cdp_captures_vectorstore()
            logger.info(f"CDP captures vectorstore created: {self.data_store.cdp_captures_vectorstore_id}")

            self.data_store.make_documentation_vectorstore()
            logger.info(f"Documentation vectorstore created: {self.data_store.documentation_vectorstore_id}")

            # Initialize and run agent
            llm_client = LLMClient(llm_model=OpenAIModel(self.llm_model))
            self.agent = RoutineDiscoveryAgent(
                llm_client=llm_client,
                data_store=self.data_store,
                task=self.task,
                emit_message_callable=self.message_callback,
                output_dir=self.output_dir,
                remote_debugging_address=self.remote_debugging_address,
            )

            # Run agent and get routine
            routine = self.agent.run()
            logger.info("Routine discovery completed successfully.")

            # Get test parameters from the agent
            test_parameters = self.agent.get_test_parameters(routine)
            logger.info("Test parameters generated successfully.")

            # Save test parameters to output directory
            test_params_dict = {p.name: p.value for p in test_parameters.parameters}
            test_params_path = Path(self.output_dir) / "test_parameters.json"
            test_params_path.write_text(json.dumps(test_params_dict, indent=2))
            logger.info(f"Test parameters saved to: {test_params_path}")

            return RoutineDiscoveryResult(
                routine=routine,
                test_parameters=test_parameters
            )

        finally:
            # Clean up vectorstore
            if self.data_store is not None and self.data_store.cdp_captures_vectorstore_id is not None:
                logger.info("Cleaning up vectorstore...")
                self.data_store.clean_up()
                logger.info("Vectorstore cleaned up.")

