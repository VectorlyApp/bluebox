"""
CDP Proxy Sidecar — sits between your service and Chrome.

Your service connects to the sidecar as if it were Chrome. All CDP messages
pass through transparently, except:

  Target.createBrowserContext  with proxyServer="user:pass@host:port"
    → sidecar spins up a local auth-forwarding proxy on a free port,
      rewrites proxyServer to 127.0.0.1:<port>, forwards to Chrome.

  Target.disposeBrowserContext
    → sidecar tears down the local proxy for that context.

Usage:
    python cdp_proxy_sidecar.py                        # defaults
    python cdp_proxy_sidecar.py --listen-port 9223     # custom listen port
    python cdp_proxy_sidecar.py --chrome-port 9222     # custom chrome port

Your service then connects to ws://127.0.0.1:9223 (or /json/version on HTTP)
instead of Chrome directly.
"""

import argparse
import asyncio
import base64
import json
import logging
from urllib.parse import urlparse

import requests
import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve, ServerConnection
from websockets.http11 import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [sidecar] %(message)s")
log = logging.getLogger("sidecar")

# ── Forward proxy (reused from earlier, handles CONNECT + plain HTTP) ────────

async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def handle_proxy_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
    proxy_auth_b64: str,
) -> None:
    try:
        first_line = await asyncio.wait_for(client_reader.readline(), timeout=30)
        if not first_line:
            client_writer.close()
            return

        method = first_line.decode("utf-8", errors="replace").strip().split(" ")[0].upper()

        headers_raw = b""
        while True:
            line = await asyncio.wait_for(client_reader.readline(), timeout=10)
            headers_raw += line
            if line == b"\r\n" or line == b"\n" or not line:
                break

        up_reader, up_writer = await asyncio.open_connection(upstream_host, upstream_port)
        auth_hdr = f"Proxy-Authorization: Basic {proxy_auth_b64}\r\n".encode()

        if method == "CONNECT":
            up_writer.write(first_line + auth_hdr + headers_raw)
            await up_writer.drain()

            up_response = await asyncio.wait_for(up_reader.readline(), timeout=30)
            while True:
                hdr_line = await asyncio.wait_for(up_reader.readline(), timeout=10)
                if hdr_line == b"\r\n" or hdr_line == b"\n" or not hdr_line:
                    break

            if b"200" in up_response:
                client_writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await client_writer.drain()
                await asyncio.gather(
                    pipe(client_reader, up_writer),
                    pipe(up_reader, client_writer),
                )
            else:
                client_writer.write(up_response + b"\r\n")
                await client_writer.drain()
                client_writer.close()
                up_writer.close()
        else:
            up_writer.write(first_line + auth_hdr + headers_raw)
            await up_writer.drain()
            await asyncio.gather(
                pipe(client_reader, up_writer),
                pipe(up_reader, client_writer),
            )
    except Exception as e:
        log.debug("proxy handler error: %s", e)
        try:
            client_writer.close()
        except Exception:
            pass


# ── ProxyManager: single event loop, dynamic ports ──────────────────────────

class ProxyManager:
    def __init__(self, port_start: int = 18001, port_count: int = 10) -> None:
        self._port_start = port_start
        self._port_count = port_count
        self._next_idx = 0
        self._servers: dict[int, asyncio.Server] = {}

    async def allocate(self, upstream_host: str, upstream_port: int, username: str, password: str) -> int:
        port = self._port_start + (self._next_idx % self._port_count)
        self._next_idx += 1

        # If port still has a lingering server, close it first
        old = self._servers.pop(port, None)
        if old:
            old.close()

        auth_b64 = base64.b64encode(f"{username}:{password}".encode()).decode()

        async def on_client(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            await handle_proxy_client(r, w, upstream_host, upstream_port, auth_b64)

        server = await asyncio.start_server(on_client, "127.0.0.1", port)
        self._servers[port] = server
        log.info("proxy port %d UP → %s:%d (user: ...%s)  [active=%d]", port, upstream_host, upstream_port, username[-20:], len(self._servers))
        return port

    async def release(self, port: int) -> None:
        server = self._servers.pop(port, None)
        if server:
            server.close()
            log.info("proxy port %d DOWN  [active=%d]", port, len(self._servers))

    async def release_all(self) -> None:
        for port in list(self._servers):
            await self.release(port)


# ── CDP WebSocket Sidecar ───────────────────────────────────────────────────

def parse_proxy_with_auth(proxy_server: str) -> tuple[str, int, str, str] | None:
    """Parse proxy address with auth → (host, port, user, pass) or None if no auth.

    Supports formats:
    - user:pass@host:port
    - http://user:pass@host:port
    """
    addr = proxy_server.strip()
    if not addr.startswith(("http://", "https://")):
        addr = f"http://{addr}"
    parsed = urlparse(addr)
    if not parsed.username or not parsed.password or not parsed.hostname or not parsed.port:
        return None
    return parsed.hostname, parsed.port, parsed.username, parsed.password


class CDPProxySidecar:
    def __init__(self, chrome_host: str, chrome_port: int, listen_port: int) -> None:
        self.chrome_host = chrome_host
        self.chrome_port = chrome_port
        self.listen_port = listen_port
        self.chrome_http = f"http://{chrome_host}:{chrome_port}"
        self._chrome_ws_url: str | None = None
        self.proxy_mgr = ProxyManager()

        # Shared state: browserContextId → local proxy port
        self._context_to_port: dict[str, int] = {}

    def _get_chrome_ws_url(self) -> str:
        """Fetch Chrome WS URL, retrying until Chrome is available."""
        if self._chrome_ws_url:
            return self._chrome_ws_url
        resp = requests.get(f"{self.chrome_http}/json/version", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        ws_url = data["webSocketDebuggerUrl"]
        # Normalize to reachable host
        parsed = urlparse(ws_url)
        fixed = parsed._replace(netloc=f"{self.chrome_host}:{self.chrome_port}")
        self._chrome_ws_url = fixed.geturl()
        return self._chrome_ws_url

    async def handle_client(self, client_ws: ServerConnection) -> None:
        """Handle one client WS connection: proxy to Chrome with interception."""
        log.info("client connected")

        try:
            chrome_ws_url = self._get_chrome_ws_url()
        except Exception as e:
            log.error("Cannot reach Chrome at %s: %s", self.chrome_http, e)
            await client_ws.close(1011, f"Chrome not reachable at {self.chrome_http}")
            return

        # Per-connection tracking
        connection_ports: set[int] = set()  # all ports allocated for this connection
        pending_creates: dict[int, int] = {}  # msg_id → local_port

        async with connect(chrome_ws_url, max_size=None, ping_interval=None, ping_timeout=None) as chrome_ws:

            async def client_to_chrome() -> None:
                try:
                    async for raw_msg in client_ws:
                        data = json.loads(raw_msg)
                        method = data.get("method", "")

                        if method == "Target.createBrowserContext":
                            params = data.get("params", {})
                            proxy_val = params.get("proxyServer", "")
                            parsed = parse_proxy_with_auth(proxy_val)
                            if parsed:
                                host, port, user, pwd = parsed
                                local_port = await self.proxy_mgr.allocate(host, port, user, pwd)
                                params["proxyServer"] = f"127.0.0.1:{local_port}"
                                pending_creates[data["id"]] = local_port
                                connection_ports.add(local_port)
                                log.info("intercepted createBrowserContext (msg %d) → local port %d", data["id"], local_port)

                        elif method == "Target.disposeBrowserContext":
                            ctx_id = data.get("params", {}).get("browserContextId", "")
                            local_port = self._context_to_port.pop(ctx_id, None)
                            if local_port:
                                await self.proxy_mgr.release(local_port)
                                connection_ports.discard(local_port)
                                log.info("intercepted disposeBrowserContext %s → released port %d", ctx_id, local_port)

                        await chrome_ws.send(json.dumps(data))
                except websockets.ConnectionClosed:
                    pass

            async def chrome_to_client() -> None:
                try:
                    async for raw_msg in chrome_ws:
                        data = json.loads(raw_msg)

                        # Track createBrowserContext responses to map contextId → port
                        msg_id = data.get("id")
                        if msg_id is not None and msg_id in pending_creates:
                            ctx_id = data.get("result", {}).get("browserContextId")
                            if ctx_id:
                                self._context_to_port[ctx_id] = pending_creates.pop(msg_id)
                                log.info("mapped context %s → port %d", ctx_id, self._context_to_port[ctx_id])

                        await client_ws.send(json.dumps(data))
                except websockets.ConnectionClosed:
                    pass

            c2c = asyncio.create_task(client_to_chrome())
            c2cl = asyncio.create_task(chrome_to_client())

            done, still_pending = await asyncio.wait(
                [c2c, c2cl], return_when=asyncio.FIRST_COMPLETED
            )
            for task in still_pending:
                task.cancel()

        # Client disconnected — release all proxy ports owned by this connection
        if connection_ports:
            log.info("client disconnected, releasing %d proxy port(s)...", len(connection_ports))
            # Also clean up context_to_port entries pointing to these ports
            stale_ctx_ids = [ctx_id for ctx_id, p in self._context_to_port.items() if p in connection_ports]
            for ctx_id in stale_ctx_ids:
                self._context_to_port.pop(ctx_id, None)
            for port in connection_ports:
                await self.proxy_mgr.release(port)
        else:
            log.info("client disconnected (no ports to release)")

    async def _process_http_request(self, connection: ServerConnection, request: websockets.http11.Request) -> Response | None:
        """Handle plain HTTP requests (e.g. /json/version) before WS upgrade."""
        if request.path.startswith("/json"):
            # Proxy to Chrome, rewrite webSocketDebuggerUrl
            try:
                resp = requests.get(f"{self.chrome_http}{request.path}", timeout=5)
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text

                # Rewrite WS URLs to point to sidecar
                if isinstance(body, dict) and "webSocketDebuggerUrl" in body:
                    body["webSocketDebuggerUrl"] = f"ws://127.0.0.1:{self.listen_port}"
                if isinstance(body, list):
                    for item in body:
                        if isinstance(item, dict):
                            for key in ("webSocketDebuggerUrl", "webSocketUrl"):
                                if key in item:
                                    # Keep the path but rewrite host:port
                                    parsed = urlparse(item[key])
                                    item[key] = parsed._replace(
                                        netloc=f"127.0.0.1:{self.listen_port}"
                                    ).geturl()

                body_bytes = json.dumps(body).encode()
                return Response(
                    resp.status_code,
                    resp.reason,
                    websockets.Headers({"Content-Type": "application/json"}),
                    body_bytes,
                )
            except Exception as e:
                return Response(502, "Bad Gateway", websockets.Headers(), str(e).encode())

        # Not an HTTP-only path → proceed with WebSocket upgrade
        return None

    async def run(self) -> None:
        log.info("Chrome expected at %s (will connect lazily on first client)", self.chrome_http)
        log.info("Listening on ws://127.0.0.1:%d", self.listen_port)
        log.info("HTTP /json/* also served on http://127.0.0.1:%d", self.listen_port)

        async with serve(
            self.handle_client,
            "0.0.0.0",
            self.listen_port,
            process_request=self._process_http_request,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        ):
            await asyncio.Future()  # run forever


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CDP proxy sidecar with dynamic auth proxy allocation")
    parser.add_argument("--chrome-host", default="127.0.0.1", help="Chrome debug host (default: 127.0.0.1)")
    parser.add_argument("--chrome-port", type=int, default=9222, help="Chrome debug port (default: 9222)")
    parser.add_argument("--listen-port", type=int, default=9223, help="Sidecar listen port (default: 9223)")
    args = parser.parse_args()

    sidecar = CDPProxySidecar(args.chrome_host, args.chrome_port, args.listen_port)
    try:
        asyncio.run(sidecar.run())
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
