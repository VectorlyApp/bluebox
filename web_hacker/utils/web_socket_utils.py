"""
web_hacker/utils/web_socket_utils.py

WebSocket utility functions for CDP communication.
"""

import itertools
import json
import time
from collections.abc import Callable
from json import JSONDecodeError

from websocket import WebSocket

# Global counter for WS message IDs - guaranteed unique per process
_msg_id_counter = itertools.count(1)


def send_cmd(ws: WebSocket, method: str, params: dict | None = None, session_id: str | None = None) -> int:
    """
    Send a CDP command over WebSocket and return the message ID.
    
    Args:
        ws: WebSocket connection.
        method: CDP method name (e.g., 'Page.navigate').
        params: Optional parameters for the method.
        session_id: Optional CDP session ID.
    
    Returns:
        The message ID used for this command.
    """
    msg_id = next(_msg_id_counter)
    msg = {"id": msg_id, "method": method}
    if params:
        msg["params"] = params
    if session_id:
        msg["sessionId"] = session_id
    ws.send(json.dumps(msg))
    return msg_id


def recv_json(ws: WebSocket, deadline: float) -> dict:
    """
    Read a single JSON message from WebSocket, skipping empty/non-JSON frames.
    
    Args:
        ws: WebSocket connection.
        deadline: Unix timestamp deadline for receiving.
    
    Returns:
        Parsed JSON message as a dictionary.
    
    Raises:
        TimeoutError: If deadline is exceeded before receiving a valid JSON message.
    """
    while time.time() < deadline:
        raw = ws.recv()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except JSONDecodeError:
            continue
    raise TimeoutError("Timed out waiting for a JSON CDP message")


def recv_until(ws: WebSocket, predicate: Callable[[dict], bool], deadline: float) -> dict:
    """
    Read messages until predicate matches.
    
    Args:
        ws: WebSocket connection.
        predicate: Callable that takes a message dict and returns True when matched.
        deadline: Unix timestamp deadline for receiving.
    
    Returns:
        The message that matched the predicate.
    
    Raises:
        TimeoutError: If deadline is exceeded before finding a matching message.
    """
    while time.time() < deadline:
        msg = recv_json(ws, deadline)
        if predicate(msg):
            return msg
    raise TimeoutError("Timed out waiting for expected CDP message")
