"""
High-level WebHacker client for easy SDK usage.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from openai import OpenAI

from ..config import Config
from ..utils.exceptions import ApiKeyNotFoundError
from .monitor import BrowserMonitor
from .discovery import RoutineDiscovery, RoutineDiscoveryResult
from .execution import RoutineExecutor
from ..data_models.routine.routine import Routine


class WebHacker:
    """
    Main SDK client for Web Hacker.

    Provides a simple, high-level interface for monitoring browsers,
    discovering routines, and executing automation.

    Example:
        >>> hacker = WebHacker(openai_api_key="sk-...")
        >>> with hacker.monitor_browser(output_dir="./captures"):
        ...     # User performs actions in browser
        ...     pass
        >>> discovery_result = hacker.discover_routine(
        ...     task="Search for flights",
        ...     cdp_captures_dir="./captures"
        ... )
        >>> routine = discovery_result.routine
        >>> test_params = discovery_result.test_parameters
        >>> result = hacker.execute_routine(
        ...     routine=routine,
        ...     parameters={"origin": "NYC", "destination": "LAX"}
        ... )
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        remote_debugging_address: str = "http://127.0.0.1:9222",
        llm_model: str = "gpt-5.1",
    ):
        """
        Initialize WebHacker client.
        
        Args:
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            remote_debugging_address: Chrome debugging server address.
            llm_model: LLM model to use for routine discovery.
        """
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ApiKeyNotFoundError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.remote_debugging_address = remote_debugging_address
        self.llm_model = llm_model
        
        self._monitor = None
        self._discovery = None
        self._executor = None
    
    def monitor_browser(
        self,
        output_dir: str = "./cdp_captures",
        url: str = "about:blank",
        incognito: bool = True,
        block_patterns: Optional[list[str]] = None,
        capture_resources: Optional[set] = None,
        **kwargs
    ) -> BrowserMonitor:
        """
        Start monitoring browser activity.
        
        Args:
            output_dir: Directory to save captured data.
            url: Initial URL to navigate to.
            incognito: Whether to use incognito mode.
            block_patterns: URL patterns to block (trackers, ads, etc.).
            capture_resources: Resource types to capture.
            **kwargs: Additional options passed to BrowserMonitor.
        
        Returns:
            BrowserMonitor instance for controlling the monitoring session.
        """
        self._monitor = BrowserMonitor(
            remote_debugging_address=self.remote_debugging_address,
            output_dir=output_dir,
            url=url,
            incognito=incognito,
            block_patterns=block_patterns,
            capture_resources=capture_resources,
            **kwargs
        )
        return self._monitor
    
    def discover_routine(
        self,
        task: str,
        cdp_captures_dir: str = "./cdp_captures",
        output_dir: str = "./routine_discovery_output",
        llm_model: Optional[str] = None,
    ) -> RoutineDiscoveryResult:
        """
        Discover a routine from captured browser data.

        Args:
            task: Description of the task to automate.
            cdp_captures_dir: Directory containing CDP captures.
            output_dir: Directory to save discovery results.
            llm_model: LLM model to use (overrides default).

        Returns:
            RoutineDiscoveryResult containing the routine and test parameters.
        """
        self._discovery = RoutineDiscovery(
            client=self.client,
            task=task,
            cdp_captures_dir=cdp_captures_dir,
            output_dir=output_dir,
            llm_model=llm_model or self.llm_model,
        )
        return self._discovery.run()
    
    def execute_routine(
        self,
        routine: Routine,
        parameters: Dict[str, Any],
        timeout: float = 180.0,
        close_tab_when_done: bool = True,
        tab_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Execute a routine with given parameters.
        
        Args:
            routine: Routine to execute.
            parameters: Parameters for the routine.
            timeout: Operation timeout in seconds.
            close_tab_when_done: Whether to close tab when finished.
            tab_id: If provided, attach to this existing tab. If None, create a new tab.
        
        Returns:
            RoutineExecutionResult with "ok" status and "result" data.
        """
        self._executor = RoutineExecutor(
            remote_debugging_address=self.remote_debugging_address,
        )
        return self._executor.execute(
            routine=routine,
            parameters=parameters,
            timeout=timeout,
            close_tab_when_done=close_tab_when_done,
            tab_id=tab_id,
        )

