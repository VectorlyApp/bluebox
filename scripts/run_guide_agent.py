#!/usr/bin/env python3
"""
scripts/run_guide_agent.py

Interactive terminal interface for the Guide Agent.
Guides users through creating web automation routines.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from openai import OpenAI

from web_hacker.agents.guide_agent import GuideAgent
from web_hacker.config import Config
from web_hacker.data_models.llms.vendors import OpenAIModel
from web_hacker.data_models.llms.interaction import (
    ChatMessageType,
    EmittedChatMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from web_hacker.routine_discovery.data_store import DiscoveryDataStore, LocalDiscoveryDataStore


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def colorize(text: str, *codes: str) -> str:
    """Apply ANSI color codes to text."""
    return "".join(codes) + text + Colors.RESET


def print_wrapped(text: str, indent: str = "  ", width: int = 80) -> None:
    """Print text with word wrapping and indentation."""
    lines = text.split("\n")
    for line in lines:
        if line.strip():
            wrapped = textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
            print(wrapped)
        else:
            print()


class TerminalGuideChat:
    """Interactive terminal chat interface for the Guide Agent."""

    BANNER = r"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║  ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ██╗  ██╗   ██╗ ║
    ║  ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██║  ╚██╗ ██╔╝ ║
    ║  ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝██║   ╚████╔╝  ║
    ║  ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██║    ╚██╔╝   ║
    ║   ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗██║    ║
    ║    ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝    ║
    ║                                                                    ║
    ║                      Guide Agent Terminal                          ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """

    WELCOME_MESSAGE = """
    Welcome! I'll help you create a web automation routine from your
    CDP (Chrome DevTools Protocol) captures.

    I'll analyze your network transactions to identify relevant API
    endpoints, required cookies, headers, and request patterns that
    can be turned into a reusable routine.

    Commands:
      • Type your message and press Enter to chat
      • Type 'quit' or 'exit' to leave
      • Type 'reset' to start a new conversation

    Links:
      • Docs: https://vectorly.app/docs
      • Console: https://console.vectorly.app
    """

    def __init__(
        self,
        llm_model: OpenAIModel | None = None,
        data_store: DiscoveryDataStore | None = None,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._pending_invocation: PendingToolInvocation | None = None
        self._streaming_started: bool = False
        self._data_store = data_store
        self._agent = GuideAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=llm_model if llm_model else OpenAIModel.GPT_5_MINI,
            data_store=data_store,
        )

    def _handle_stream_chunk(self, chunk: str) -> None:
        """
        Handle a streaming text chunk from the LLM.

        Args:
            chunk: A text chunk from the streaming response.
        """
        if not self._streaming_started:
            # Print the header before the first chunk
            print()
            print(colorize("  Assistant", Colors.BOLD, Colors.CYAN) + colorize(":", Colors.DIM))
            print()
            print("    ", end="", flush=True)
            self._streaming_started = True

        # Print chunk without newline, flush immediately
        print(chunk, end="", flush=True)

    def _handle_message(self, message: EmittedChatMessage) -> None:
        """
        Handle messages emitted by the Guide Agent.

        Args:
            message: The emitted message from the agent.
        """
        if message.type == ChatMessageType.CHAT_RESPONSE:
            # If we were streaming, just finish with newlines (content already printed)
            if self._streaming_started:
                print()  # End the streamed line
                print()  # Add spacing
                self._streaming_started = False
            else:
                self._print_assistant_message(message.content or "")

        elif message.type == ChatMessageType.TOOL_INVOCATION_REQUEST:
            if message.tool_invocation:
                self._pending_invocation = message.tool_invocation
                self._print_tool_request(message.tool_invocation)

        elif message.type == ChatMessageType.TOOL_INVOCATION_RESULT:
            if message.tool_invocation:
                self._print_tool_result(
                    message.tool_invocation,
                    message.tool_result,
                    message.error,
                )

        elif message.type == ChatMessageType.ERROR:
            self._print_error(message.error or "Unknown error")

    def _print_assistant_message(self, content: str) -> None:
        """Print an assistant response."""
        print()
        print(colorize("  Assistant", Colors.BOLD, Colors.CYAN) + colorize(":", Colors.DIM))
        print()
        print_wrapped(content, indent="    ")
        print()

    def _print_tool_request(self, invocation: PendingToolInvocation) -> None:
        """Print a tool invocation request with formatted arguments."""
        print()
        print(colorize("  ┌─────────────────────────────────────────────────────────────────┐", Colors.YELLOW))
        print(colorize("  │", Colors.YELLOW) + colorize("  TOOL INVOCATION REQUEST", Colors.BOLD, Colors.YELLOW) + colorize("                                       │", Colors.YELLOW))
        print(colorize("  ├─────────────────────────────────────────────────────────────────┤", Colors.YELLOW))
        print(colorize("  │", Colors.YELLOW))

        # Tool name
        print(colorize("  │  ", Colors.YELLOW) + colorize("Tool: ", Colors.DIM) + colorize(invocation.tool_name, Colors.BRIGHT_WHITE, Colors.BOLD))

        # Arguments
        print(colorize("  │", Colors.YELLOW))
        print(colorize("  │  ", Colors.YELLOW) + colorize("Arguments:", Colors.DIM))

        args_json = json.dumps(invocation.tool_arguments, indent=4)
        for line in args_json.split("\n"):
            print(colorize("  │    ", Colors.YELLOW) + colorize(line, Colors.WHITE))

        print(colorize("  │", Colors.YELLOW))
        print(colorize("  └─────────────────────────────────────────────────────────────────┘", Colors.YELLOW))
        print()
        print(colorize("  Do you want to proceed? ", Colors.BRIGHT_YELLOW) + colorize("[y/n]", Colors.DIM) + ": ", end="")

    def _print_tool_result(
        self,
        invocation: PendingToolInvocation,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        """Print a tool invocation result."""
        print()

        if invocation.status == ToolInvocationStatus.DENIED:
            print(colorize("  ✗ Tool invocation denied", Colors.YELLOW))

        elif invocation.status == ToolInvocationStatus.EXECUTED:
            print(colorize("  ✓ Tool executed successfully", Colors.GREEN, Colors.BOLD))
            if result:
                print()
                print(colorize("  Result:", Colors.DIM))
                result_json = json.dumps(result, indent=4)
                for line in result_json.split("\n"):
                    print(colorize("    " + line, Colors.GREEN))

        elif invocation.status == ToolInvocationStatus.FAILED:
            print(colorize("  ✗ Tool execution failed", Colors.RED, Colors.BOLD))
            if error:
                print(colorize(f"    Error: {error}", Colors.RED))

        print()

    def _print_error(self, error: str) -> None:
        """Print an error message."""
        print()
        print(colorize("  ⚠ Error: ", Colors.RED, Colors.BOLD) + colorize(error, Colors.RED))
        print()

    def _print_user_prompt(self) -> None:
        """Print the user input prompt."""
        print(colorize("  You", Colors.BOLD, Colors.GREEN) + colorize(": ", Colors.DIM), end="")

    def _handle_tool_confirmation(self, user_input: str) -> bool:
        """
        Handle yes/no confirmation for pending tool invocation.

        Args:
            user_input: The user's input.

        Returns:
            True if the confirmation was handled, False otherwise.
        """
        if not self._pending_invocation:
            return False

        normalized = user_input.strip().lower()

        if normalized in ("y", "yes"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            self._agent.confirm_tool_invocation(invocation_id)
            return True

        elif normalized in ("n", "no"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            self._agent.deny_tool_invocation(invocation_id, reason="User declined")
            return True

        else:
            print(colorize("  Please enter 'y' or 'n': ", Colors.YELLOW), end="")
            return True  # Still in confirmation mode

    def run(self) -> None:
        """Run the interactive chat loop."""
        # Print banner and welcome
        print(colorize(self.BANNER, Colors.BRIGHT_MAGENTA, Colors.BOLD))
        print(colorize(self.WELCOME_MESSAGE, Colors.DIM))
        print(colorize(f"    Model: {self._agent.llm_model}", Colors.DIM))
        print()
        print(colorize("  " + "─" * 67, Colors.DIM))
        print()

        while True:
            try:
                # Handle pending tool confirmation
                if self._pending_invocation:
                    user_input = input()
                    if self._handle_tool_confirmation(user_input):
                        if not self._pending_invocation:
                            # Confirmation was processed, continue to next iteration
                            continue
                        else:
                            # Still waiting for valid y/n
                            continue
                else:
                    self._print_user_prompt()
                    user_input = input()

                # Check for commands
                normalized = user_input.strip().lower()

                if normalized in ("quit", "exit", "q"):
                    print()
                    print(colorize("  Goodbye! 👋", Colors.CYAN, Colors.BOLD))
                    print()
                    break

                if normalized == "reset":
                    self._agent.reset()
                    self._pending_invocation = None
                    print()
                    print(colorize("  ↺ Conversation reset", Colors.YELLOW))
                    print()
                    continue

                if not user_input.strip():
                    continue

                # Process the message
                self._agent.process_user_message(user_input)

            except KeyboardInterrupt:
                print()
                print(colorize("\n  Interrupted. Goodbye! 👋", Colors.CYAN))
                print()
                break

            except EOFError:
                print()
                print(colorize("\n  Goodbye! 👋", Colors.CYAN))
                print()
                break


def parse_model(model_str: str) -> OpenAIModel:
    """Parse a model string into an OpenAIModel enum value."""
    for model in OpenAIModel:
        if model.value == model_str or model.name == model_str:
            return model
    raise ValueError(f"Unknown model: {model_str}")


def main() -> None:
    """Entry point for the guide agent terminal."""
    parser = argparse.ArgumentParser(description="Interactive Guide Agent terminal")
    parser.add_argument(
        "--model",
        type=str,
        default=OpenAIModel.GPT_5_MINI.value,
        help=f"LLM model to use (default: {OpenAIModel.GPT_5_MINI.value})",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=str,
        default=None,
        help="Path to CDP captures directory (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./guide_agent_output",
        help="Output directory for temporary files (default: ./guide_agent_output)",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="./agent_docs",
        help="Documentation directory (default: ./agent_docs)",
    )
    args = parser.parse_args()

    # Validate API key
    if Config.OPENAI_API_KEY is None:
        print(colorize("\n  Error: OPENAI_API_KEY environment variable is not set", Colors.RED, Colors.BOLD), file=sys.stderr)
        sys.exit(1)

    data_store: LocalDiscoveryDataStore | None = None

    try:
        llm_model = parse_model(args.model)

        # Initialize OpenAI client for data store
        openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

        # Build data store kwargs - CDP paths are optional
        data_store_kwargs: dict[str, Any] = {
            "client": openai_client,
            "documentation_dirs": [args.docs_dir],
            "code_dirs": ["./web_hacker/data_models"],
        }

        # Add CDP captures paths if provided
        if args.cdp_captures_dir:
            data_store_kwargs.update({
                "tmp_dir": str(Path(args.output_dir) / "tmp"),
                "transactions_dir": str(Path(args.cdp_captures_dir) / "network" / "transactions"),
                "consolidated_transactions_path": str(Path(args.cdp_captures_dir) / "network" / "consolidated_transactions.json"),
                "storage_jsonl_path": str(Path(args.cdp_captures_dir) / "storage" / "events.jsonl"),
                "window_properties_path": str(Path(args.cdp_captures_dir) / "window_properties" / "window_properties.json"),
            })

        # Create data store
        print(colorize("  Initializing data store...", Colors.DIM))
        data_store = LocalDiscoveryDataStore(**data_store_kwargs)

        # Create vectorstores
        if args.cdp_captures_dir:
            print(colorize("  Creating CDP captures vectorstore...", Colors.DIM))
            data_store.make_cdp_captures_vectorstore()
        print(colorize("  Creating documentation vectorstore...", Colors.DIM))
        data_store.make_documentation_vectorstore()
        print(colorize("  Vectorstores ready!", Colors.GREEN))
        print()

        chat = TerminalGuideChat(llm_model=llm_model, data_store=data_store)
        chat.run()

    except ValueError as e:
        print(colorize(f"\n  Error: {e}", Colors.RED, Colors.BOLD), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(colorize(f"\n  Fatal error: {e}", Colors.RED, Colors.BOLD), file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up vectorstores
        if data_store is not None:
            print(colorize("\n  Cleaning up vectorstores...", Colors.DIM))
            try:
                data_store.clean_up()
                print(colorize("  Cleanup complete!", Colors.GREEN))
            except Exception as e:
                print(colorize(f"  Warning: Cleanup failed: {e}", Colors.YELLOW), file=sys.stderr)


if __name__ == "__main__":
    main()
