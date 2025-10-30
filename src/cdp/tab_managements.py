import json
import time
import requests
import websocket


def cdp_new_tab(
    remote_debugging_address: str = "http://127.0.0.1:9222",
    incognito: bool = True,
    url: str = "about:blank"
) -> tuple[str, str]:
    """
    Create a new tab using the Chrome DevTools Protocol.

    Args:
        remote_debugging_address (str): The address of the remote debugging server.
        incognito (bool): Whether to create an incognito context.
        url (str): The URL to navigate to.

    Returns:
        tuple: A tuple containing the target ID and the browser context ID.
    """
    base = remote_debugging_address.rstrip('/')

    # 1) Get the browser WS endpoint
    ver = requests.get(f"{base}/json/version", timeout=5)
    ver.raise_for_status()
    ws_url = ver.json()["webSocketDebuggerUrl"]

    ws = websocket.create_connection(ws_url, timeout=10)
    browser_context_id = None
    try:
        next_id = 1
        def send(method, params=None):
            nonlocal next_id
            msg = {"id": next_id, "method": method}
            if params: msg["params"] = params
            ws.send(json.dumps(msg))
            next_id += 1
            return msg["id"]

        def recv(id_):
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == id_:
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    return msg["result"]

        # 2) Create an incognito context
        if incognito:
            iid = send("Target.createBrowserContext")
            browser_context_id = recv(iid)["browserContextId"]

        # 3) Create the target and force a NEW WINDOW if we're incognito
        params = {"url": url}
        if browser_context_id:
            params["browserContextId"] = browser_context_id
            params["newWindow"] = True

        tid = send("Target.createTarget", params)
        target_id = recv(tid)["targetId"]

        return target_id, browser_context_id

    finally:
        try: ws.close()
        except: pass


def _navigate_to_url(
    send_cmd: callable,
    recv_json: callable,
    ws: websocket.WebSocket,
    url: str,
    session_id: str,
    timeout: float
) -> None:
    """
    Navigate to a URL using the Chrome DevTools Protocol.

    Args:
        send_cmd (callable): A function to send a command to the Chrome DevTools Protocol.
        recv_json (callable): A function to receive a JSON message from the Chrome DevTools Protocol.
        ws (websocket.WebSocket): The WebSocket connection to the Chrome DevTools Protocol.
        url (str): The URL to navigate to.
        session_id (str): The session ID.
        timeout (float): The timeout in seconds.
    """
    
    send_cmd("Page.navigate", {"url": url}, session_id=session_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = recv_json(ws, deadline)
        if msg.get("sessionId") != session_id:
            continue
        if msg.get("method") in ("Page.loadEventFired", "Runtime.executionContextCreated"):
            break
        
        
def dispose_context(
    remote_debugging_address: str,
    browser_context_id: str
) -> None:
    """
    Dispose a browser context using the Chrome DevTools Protocol.

    Args:
        remote_debugging_address (str): The address of the remote debugging server.
        browser_context_id (str): The browser context ID.
    """
    
    base = remote_debugging_address.rstrip('/')
    ver = requests.get(f"{base}/json/version", timeout=5)
    ver.raise_for_status()
    ws_url = ver.json()["webSocketDebuggerUrl"]
    ws = websocket.create_connection(ws_url, timeout=10)
    try:
        ws.send(json.dumps({"id": 1, "method": "Target.disposeBrowserContext",
                        "params": {"browserContextId": browser_context_id}}))
        json.loads(ws.recv())
    finally:
        try: ws.close()
        except: pass