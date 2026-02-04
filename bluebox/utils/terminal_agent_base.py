"""
bluebox/utils/terminal_agent_base.py

Abstract base class for terminal-based agent chat interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich import box
from rich.console import Console
from rich.panel import Panel

from bluebox.data_models.llms.interaction import (
    ChatRole,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
)
from bluebox.utils.terminal_utils import (
    SlashCommandCompleter,
    SlashCommandLexer,
    print_assistant_message,
    print_error,
    print_tool_call,
    print_tool_result,
)


class AbstractTerminalAgentChat(ABC):
    """
    Abstract base class for terminal-based agent chat interfaces.

    This class provides the common infrastructure for interactive terminal chat
    with AI agents, including:
    - Message handling and streaming
    - Interactive command loop with slash commands
    - Standard commands (/quit, /reset, /help)
    - Autonomous command handling

    Subclasses must implement:
    - _create_agent(): Create the specific agent instance
    - get_slash_commands(): Return list of slash commands
    - print_welcome(): Print agent-specific welcome screen
    - handle_autonomous_command(): Handle autonomous command execution
    - autonomous_command_name: Property returning the autonomous command name
    """

    def __init__(self, console: Console, agent_color: str) -> None:
        """
        Initialize the terminal chat interface.

        Args:
            console: Rich Console instance for output
            agent_color: Rich color name for agent-specific styling (e.g., "cyan", "green")
        """
        self.console = console
        self.agent_color = agent_color
        self._streaming_started: bool = False
        self._agent = self._create_agent()

    @abstractmethod
    def _create_agent(self) -> Any:
        """
        Create and return the agent instance.

        The agent should be initialized with:
        - emit_message_callable=self._handle_message
        - stream_chunk_callable=self._handle_stream_chunk

        Returns:
            The agent instance
        """

    @abstractmethod
    def get_slash_commands(self) -> list[tuple[str, str]]:
        """
        Return list of slash commands for this agent.

        Returns:
            List of (command, description) tuples, e.g.:
            [
                ("/discover", "Discover API endpoints"),
                ("/reset", "Start a new conversation"),
                ("/help", "Show help"),
                ("/quit", "Exit"),
            ]
        """

    @abstractmethod
    def print_welcome(self) -> None:
        """
        Print the agent-specific welcome message with stats/info.

        This is typically called once before starting the interactive loop.
        """

    @abstractmethod
    def handle_autonomous_command(self, task: str) -> None:
        """
        Handle execution of the autonomous command.

        Args:
            task: The task description provided by the user

        Example:
            For /discover command: "train prices from NYC to Boston"
            For /trace command: "eyJhbGciOiJIUzI1NiJ9"
        """

    @property
    @abstractmethod
    def autonomous_command_name(self) -> str:
        """
        Return the autonomous command name (without leading slash).

        Returns:
            Command name, e.g.: "discover", "trace", "search", "autonomous"
        """

    def _handle_stream_chunk(self, chunk: str) -> None:
        """
        Handle streaming text chunks from the LLM.

        Args:
            chunk: A chunk of text to stream to the terminal
        """
        if not self._streaming_started:
            self.console.print()
            self.console.print(f"[bold {self.agent_color}]Assistant[/bold {self.agent_color}]")
            self.console.print()
            self._streaming_started = True

        print(chunk, end="", flush=True)

    def _handle_message(self, message: EmittedMessage) -> None:
        """
        Handle messages emitted by the agent.

        Args:
            message: The emitted message from the agent
        """
        if isinstance(message, ChatResponseEmittedMessage):
            if self._streaming_started:
                # Streaming just finished
                print()
                print()
                self._streaming_started = False
            else:
                # Non-streaming response
                print_assistant_message(message.content, self.console)

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            # Show tool call and result
            print_tool_call(message.tool_invocation, self.console)
            print_tool_result(message.tool_invocation, message.tool_result, self.console)

        elif isinstance(message, ErrorEmittedMessage):
            print_error(message.error, self.console)

    def run(self) -> None:
        """
        Run the interactive chat loop.

        Handles user input, slash commands, and delegates to the agent.
        """
        while True:
            try:
                user_input = pt_prompt(
                    HTML(f"<b><ansi{self.agent_color}>You&gt;</ansi{self.agent_color}></b> "),
                    completer=SlashCommandCompleter(self.get_slash_commands()),
                    lexer=SlashCommandLexer(),
                    complete_while_typing=True,
                )

                if not user_input.strip():
                    continue

                cmd = user_input.strip().lower()

                # Handle quit
                if cmd in ("/quit", "/exit", "/q"):
                    self.console.print()
                    self.console.print(f"[bold {self.agent_color}]Goodbye![/bold {self.agent_color}]")
                    self.console.print()
                    break

                # Handle reset
                if cmd == "/reset":
                    self._agent.reset()
                    self.console.print()
                    self.console.print("[yellow]Conversation reset[/yellow]")
                    self.console.print()
                    continue

                # Handle help
                if cmd in ("/help", "/h", "/?"):
                    self._show_help()
                    continue

                # Handle autonomous command
                if user_input.strip().lower().startswith(f"/{self.autonomous_command_name}"):
                    task = user_input.strip()[len(f"/{self.autonomous_command_name}"):].strip()
                    if not task:
                        self.console.print()
                        self.console.print(f"[bold yellow]Usage:[/bold yellow] /{self.autonomous_command_name} <task>")
                        self.console.print()
                        continue
                    self.handle_autonomous_command(task)
                    continue

                # Normal message - send to agent
                self._agent.process_new_message(user_input, ChatRole.USER)

            except KeyboardInterrupt:
                self.console.print()
                self.console.print(f"[{self.agent_color}]Interrupted. Goodbye![/{self.agent_color}]")
                self.console.print()
                break

            except EOFError:
                self.console.print()
                self.console.print(f"[{self.agent_color}]Goodbye![/{self.agent_color}]")
                self.console.print()
                break

    def _show_help(self) -> None:
        """Show help panel with available commands."""
        # Build help text from slash commands
        commands_text = "\n".join(
            f"  [{self.agent_color}]{cmd}[/{self.agent_color}]  {desc}"
            for cmd, desc in self.get_slash_commands()
        )

        self.console.print()
        self.console.print(Panel(
            f"[bold]Commands:[/bold]\n{commands_text}",
            title=f"[bold {self.agent_color}]Help[/bold {self.agent_color}]",
            border_style=self.agent_color,
            box=box.ROUNDED,
        ))
        self.console.print()
