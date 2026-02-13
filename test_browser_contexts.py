"""
Test script for the CDP proxy sidecar.

1. Start the sidecar:   python cdp_proxy_sidecar.py
2. Run this test:        python test_browser_contexts.py

This script connects to the SIDECAR (not Chrome directly) and sends
Target.createBrowserContext with proxyServer="user:pass@host:port".
The sidecar transparently handles auth proxy allocation.

Press Ctrl+C to tear down.
"""

import json
import signal
import time
from dataclasses import dataclass

import requests
import websocket

# ── Configuration ────────────────────────────────────────────────────────────
SIDECAR_URL = "http://127.0.0.1:9223"

TEST_URL = "https://geo.brdtest.com/mygeo.json"

PERSIST_SECONDS = 5 * 60  # keep browsers alive for 5 minutes

# Rotating proxies — format: host:port:user:pass
ROTATING_PROXIES = [
    "68.182.104.202:56870:buykvgtw:t669WYCuRalP",
    "68.182.104.122:49653:buykvgtw:t669WYCuRalP",
]

NUM_TABS = 2
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ContextInfo:
    label: str
    context_id: str
    target_id: str


def get_ws_url(http_url: str) -> str:
    resp = requests.get(f"{http_url}/json/version", timeout=5)
    resp.raise_for_status()
    return resp.json()["webSocketDebuggerUrl"]


def send_cmd(ws: websocket.WebSocket, method: str, params: dict | None = None, *, _counter: list[int] = [0]) -> int:
    _counter[0] += 1
    msg = {"id": _counter[0], "method": method}
    if params:
        msg["params"] = params
    ws.send(json.dumps(msg))
    return _counter[0]


def recv_until(ws: websocket.WebSocket, msg_id: int, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = ws.recv()
        if not raw:
            continue
        data = json.loads(raw)
        if data.get("id") == msg_id:
            return data
    raise TimeoutError(f"Timed out waiting for response to message {msg_id}")


def parse_proxy_string(proxy: str) -> str:
    """Convert 'host:port:user:pass' → 'user:pass@host:port' for the sidecar."""
    parts = proxy.split(":")
    host, port, user, pwd = parts[0], parts[1], parts[2], parts[3]
    return f"{user}:{pwd}@{host}:{port}"


def create_context_with_tab(ws: websocket.WebSocket, proxy_server: str, url: str, label: str) -> ContextInfo:
    # The sidecar intercepts this and rewrites proxyServer if it has auth
    mid = send_cmd(ws, "Target.createBrowserContext", {
        "proxyServer": proxy_server,
        "disposeOnDetach": True,
    })
    reply = recv_until(ws, mid)
    if "error" in reply:
        raise RuntimeError(f"[{label}] createBrowserContext failed: {reply['error']}")
    context_id = reply["result"]["browserContextId"]
    print(f"[{label}] Created browser context {context_id}")

    mid = send_cmd(ws, "Target.createTarget", {
        "url": url,
        "browserContextId": context_id,
    })
    reply = recv_until(ws, mid)
    if "error" in reply:
        raise RuntimeError(f"[{label}] createTarget failed: {reply['error']}")
    target_id = reply["result"]["targetId"]
    print(f"[{label}] Opened tab -> {url}")

    return ContextInfo(label=label, context_id=context_id, target_id=target_id)


def dispose_context(ws: websocket.WebSocket, ctx: ContextInfo) -> None:
    try:
        mid = send_cmd(ws, "Target.disposeBrowserContext", {"browserContextId": ctx.context_id})
        recv_until(ws, mid, timeout=5)
        print(f"[{ctx.label}] Disposed")
    except Exception as e:
        print(f"[{ctx.label}] Warning: {e}")


def main() -> None:
    ws_url = get_ws_url(SIDECAR_URL)
    print(f"Connecting to sidecar at {ws_url}")
    ws = websocket.create_connection(ws_url, timeout=15)
    print("Connected.\n")

    contexts: list[ContextInfo] = []
    for i in range(NUM_TABS):
        proxy_raw = ROTATING_PROXIES[i % len(ROTATING_PROXIES)]
        proxy_server = parse_proxy_string(proxy_raw)
        label = f"CTX-{i + 1}"
        ctx = create_context_with_tab(ws, proxy_server, TEST_URL, label)
        contexts.append(ctx)

    print(f"\n{len(contexts)} tabs live. Persisting for {PERSIST_SECONDS // 60} minutes (Ctrl+C to stop early).\n")

    stop = False

    def on_sigint(sig: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_sigint)

    deadline = time.time() + PERSIST_SECONDS
    while not stop and time.time() < deadline:
        remaining = int(deadline - time.time())
        mins, secs = divmod(remaining, 60)
        print(f"\r  Time remaining: {mins}m {secs:02d}s  ", end="", flush=True)
        time.sleep(1)

    print("\n\nCleaning up...")
    for ctx in contexts:
        dispose_context(ws, ctx)
    ws.close()
    print("Done.")


if __name__ == "__main__":
    main()
