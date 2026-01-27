"""
bluebox/cdp/network_monitor.py

Network monitoring via CDP — *minimal*, non‑blocking, and reliable.

What this version does:
- Auto‑attaches to all targets (like DevTools).
- Intercepts **only** the problem endpoint at the **Response** stage using `Fetch`.
- Uses **Fetch.getResponseBody** (not streaming) to *tee* the body while the
  request is paused, then resumes with `Fetch.continueResponse`.
- Uses `Network.getResponseBody` after `loadingFinished` for your chosen resource
  types (unchanged behavior for everything else).
- Wraps the pause flow in try/except so a failure never leaves the page paused.

Why this fixes "200 but empty body":
- `Fetch.takeResponseBodyAsStream` can *drain* the response; unless you then
  manually feed the body back, the page gets an empty payload. `Fetch.getResponseBody`
  reads a copy of the paused body and the page still receives it when we call
  `Fetch.continueResponse`.
"""

import os
import base64
import time
import json
from datetime import datetime
from fnmatch import fnmatch
from typing import Any

from bluebox.config import Config
from bluebox.utils.data_utils import write_json_file, write_jsonl
from bluebox.utils.data_utils import build_transaction_dir, get_set_cookie_values
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


class NetworkMonitor:
    """
    Network monitor for CDP.
    """

    def __init__(self, output_dir, paths, capture_resources=None, block_patterns=None):
        self.output_dir = output_dir
        self.paths = paths
        self.capture_resources = capture_resources or set()
        self.block_patterns = block_patterns or []

        self.req_meta: dict[str, dict[str, Any]] = {}
        self.pending_body: dict[int, str] = {}
        self.fetch_get_body_wait: dict[int, dict[str, Any]] = {}
        self.session_registry: dict[int, Any] = {}  # session_id -> CDPSession
        
    # ------------------------ Setup ------------------------
    def setup_network_monitoring(self, cdp_session):
        # Attach to ALL targets (workers/iframes) like DevTools
        cdp_session.send("Target.setAutoAttach", {
            "autoAttach": True,
            "waitForDebuggerOnStart": False,
            "flatten": True,
        })

        cdp_session.send("Network.enable", {
            "includeExtraInfo": True,
            "maxTotalBufferSize": 512_000_000,
            "maxResourceBufferSize": 256_000_000,
        })
        cdp_session.send("Network.setCacheDisabled", {"cacheDisabled": True})
        cdp_session.send("Network.setBypassServiceWorker", {"bypass": True})

        if self.block_patterns:
            cdp_session.send("Network.setBlockedURLs", {"urls": self.block_patterns})

        # Enable fetch interception for both request and response stages
        # Request stage: for blocking unwanted URLs
        # Response stage: for capturing response bodies
        cdp_session.send("Fetch.enable", {
            "patterns": [
                {"urlPattern": "*", "requestStage": "Request"},
                {"urlPattern": "*", "requestStage": "Response"},
            ]
        })

        cdp_session.send("Page.enable", {})  # optional: just for logs

    # ------------------------ Dispatch ------------------------
    def handle_network_message(self, msg, cdp_session):
        method = msg.get("method")
        if method == "Fetch.requestPaused":
            return self._on_fetch_request_paused(msg, cdp_session)
        if method == "Network.requestWillBeSent":
            return self._on_request_will_be_sent(msg, cdp_session)
        if method == "Network.responseReceived":
            return self._on_response_received(msg)
        if method == "Network.loadingFinished":
            return self._on_loading_finished(msg, cdp_session)
        if method == "Network.loadingFailed":
            return self._on_loading_failed(msg)
        if method == "Network.responseReceivedExtraInfo":
            return self._on_response_received_extra_info(msg)
        if method == "Page.frameNavigated":
            try:
                url = msg["params"]["frame"].get("url")
                # Skip logging for about:blank to reduce noise
                if url and url != "about:blank":
                    logger.info("Frame navigated to: %s", url)
            except Exception:
                pass
            return False # Don't swallow this event
        return False

    def handle_network_command_reply(self, msg, cdp_session):
        self.session_registry[id(cdp_session)] = cdp_session
        cmd_id = msg.get("id")
        if cmd_id is None:
            return False

        if cmd_id in self.pending_body:
            self._on_get_response_body_reply(cmd_id, msg, cdp_session)
            return True
        if cmd_id in self.fetch_get_body_wait:
            self._on_fetch_get_body_reply(cmd_id, msg, cdp_session)
            return True
        return False

    # ------------------------ Event Handlers ------------------------
    def _on_fetch_request_paused(self, msg, cdp_session):
        p = msg["params"]
        rid = p["requestId"]
        url = ((p.get("request") or {}).get("url") or "")

        # Check if URL should be blocked
        if self._should_block_url(url):
            try:
                cdp_session.send("Fetch.failRequest", {"requestId": rid, "errorReason": "Blocked"})
                logger.info("Blocked request: %s", url)
                return True
            except Exception as e:
                logger.info("Failed to block request %s: %s", url, e)
                # Fall through to continue normally if blocking fails

        # Only our target URL at RESPONSE stage
        if p.get("responseStatusCode") is not None:
            # Check if this request type should be captured
            network_id = p.get("networkId") or rid
            req_meta = self.req_meta.get(network_id, {})
            resource_type = req_meta.get("type")
            
            # Only capture response bodies for resources matching self.capture_resources
            if resource_type in self.capture_resources:
                raw_hdrs = (p.get("responseHeaders") or [])
                hdrs = {h["name"].lower(): h["value"] for h in raw_hdrs}
                transaction_dir = req_meta.get("transactionDir")
                
                if transaction_dir:
                    # Get proper extension for response body
                    ct = hdrs.get("content-type", "").lower()
                    ext = self._guess_extension(ct) or ".bin"
                    body_path = os.path.join(transaction_dir, f"response_body{ext}")
                    
                    # Prefer Fetch.getResponseBody while paused (does not drain the body)
                    try:
                        rb_id = cdp_session.send("Fetch.getResponseBody", {"requestId": rid})
                        self.fetch_get_body_wait[rb_id] = {
                            "rid": rid,
                            "url": url,
                            "headers": hdrs,
                            "status": p.get("responseStatusCode"),
                            "body_path": body_path,
                            "transaction_dir": transaction_dir,
                            "_sess_id": id(cdp_session),
                        }
                        return True
                    except Exception as e:
                        # If we can't fetch body, do NOT block the page.
                        logger.info("Fetch.getResponseBody failed for %s: %s", url, e)
                        self._safe_continue_response(cdp_session, rid)
                        return True
            
            # Continue response immediately for non-captured resource types
            self._safe_continue_response(cdp_session, rid)
            return True

        # Everything else → resume immediately so nothing stalls
        if p.get("responseStatusCode") is not None:
            self._safe_continue_response(cdp_session, rid)
        else:
            # For request stage, also check if URL should be blocked
            if self._should_block_url(url):
                try:
                    cdp_session.send("Fetch.failRequest", {"requestId": rid, "errorReason": "Blocked"})
                    logger.info("Blocked request: %s", url)
                    return True
                except Exception as e:
                    logger.info("Failed to block request %s: %s", url, e)
                    
            self._safe_continue_request(cdp_session, rid)
        return True

    def _on_request_will_be_sent(self, msg, cdp_session):
        p = msg["params"]
        request_id = p["requestId"]
        self.req_meta[request_id] = {
            "url": p["request"]["url"],
            "method": p["request"]["method"],
            "type": p.get("type"),
            "ts": p.get("timestamp"),
            "initiator": p.get("initiator"),
            "requestHeaders": p["request"].get("headers", {}),
            "postData": p["request"].get("postData"),
            "_sess_id": id(cdp_session),
        }
        
        if self.req_meta[request_id]["type"] in self.capture_resources:
            ts_ms = int(time.time() * 1000)
            transactions_dir = self.paths.get("transactions_dir", os.path.join(self.output_dir, "transactions"))
            transaction_dir = build_transaction_dir(self.req_meta[request_id]["url"], ts_ms, transactions_dir)
            self.req_meta[request_id]["transactionDir"] = transaction_dir
            
            # Create clean request data - only essential fields
            clean_request_data = {
                "requestId": request_id,
                "url": self.req_meta[request_id]["url"],
                "method": self.req_meta[request_id]["method"],
                "type": self.req_meta[request_id]["type"],
                "requestHeaders": self.req_meta[request_id]["requestHeaders"],
            }
            
            # Only include postData if it exists
            if self.req_meta[request_id].get("postData"):
                clean_request_data["postData"] = self.req_meta[request_id]["postData"]
                
            write_json_file(os.path.join(transaction_dir, "request.json"), clean_request_data)
        return True

    def _on_response_received(self, msg):
        p = msg["params"]
        m = self.req_meta.setdefault(p["requestId"], {})
        resp = p["response"]
        m.update({
            "status": resp.get("status"),
            "statusText": resp.get("statusText"),
            "responseHeaders": resp.get("headers", {}),
            "mimeType": resp.get("mimeType"),
            "remoteIPAddress": resp.get("remoteIPAddress"),
            "remotePort": resp.get("remotePort"),
            "fromDiskCache": resp.get("fromDiskCache"),
            "fromPrefetchCache": resp.get("fromPrefetchCache"),
            "fromServiceWorker": resp.get("fromServiceWorker"),
        })
        set_cookie_values = get_set_cookie_values(resp.get("headers", {}))
        if set_cookie_values:
            existing = m.setdefault("setCookies", [])
            existing.extend(set_cookie_values)
            
            # Log cookies to cookies file
            try:
                for cookie in set_cookie_values:
                    cookie_log = {
                        "timestamp": time.time(),
                        "type": "cookie",
                        "requestId": p["requestId"],
                        "url": m.get("url", "unknown"),
                        "cookie": cookie
                    }
                    write_jsonl(self.cookies_log_path, cookie_log)
            except Exception as e:
                logger.info("Warning: Could not log cookie: %s", e)
            
        return True

    def _on_loading_finished(self, msg, cdp_session):
        request_id = msg["params"]["requestId"]
        meta = self.req_meta.get(request_id, {})
        if meta.get("type") not in self.capture_resources:
            return True
        sess_id = meta.get("_sess_id")
        sess = self.session_registry.get(sess_id, cdp_session) if sess_id else cdp_session
        try:
            cmd_id = sess.send("Network.getResponseBody", {"requestId": request_id})
            self.pending_body[cmd_id] = request_id
        except Exception as e:
            # Error getting response body - could log this elsewhere if needed
            pass
        return True

    def _on_loading_failed(self, msg):
        p = msg["params"]
        m = self.req_meta.setdefault(p["requestId"], {})
        m["failed"] = True
        m["errorText"] = p.get("errorText")
        resource_type = m.get("type") or p.get("type")
        # Loading failed - could log this elsewhere if needed
        return True

    def _on_response_received_extra_info(self, msg):
        p = msg["params"]
        request_id = p.get("requestId")
        m = self.req_meta.setdefault(request_id, {})
        set_cookie_values = get_set_cookie_values(p.get("headers", {}))
        if set_cookie_values and not m.get("cookiesLogged"):
            m["setCookies"] = set_cookie_values
            m["cookiesLogged"] = True
        return True

    # ------------------------ Command Replies ------------------------
    def _on_get_response_body_reply(self, cmd_id, msg, cdp_session):
        request_id = self.pending_body.pop(cmd_id)
        req_meta = self.req_meta.get(request_id, {})
        url = req_meta.get("url", "unknown")

        if "error" in msg:
            # Error getting response body - already handled in transaction directory
            return

        body_info = msg.get("result", {})
        data = body_info.get("body", "")
        is_b64 = body_info.get("base64Encoded", False)
        if not data:
            # Only save response-specific info for empty body
            response_info = {
                "url": url,
                "status": req_meta.get("status"),
                "statusText": req_meta.get("statusText"),
                "responseHeaders": req_meta.get("responseHeaders", {}),
                "mimeType": req_meta.get("mimeType"),
                "note": "empty-body"
            }
            # Empty body case - data already saved in transaction directory
            return

        # Create response-only info object
        response_info = {
            "url": url,
            "status": req_meta.get("status"),
            "statusText": req_meta.get("statusText"),
            "responseHeaders": req_meta.get("responseHeaders", {}),
            "mimeType": req_meta.get("mimeType"),
            "remoteIPAddress": req_meta.get("remoteIPAddress"),
            "remotePort": req_meta.get("remotePort"),
            "fromDiskCache": req_meta.get("fromDiskCache"),
            "fromPrefetchCache": req_meta.get("fromPrefetchCache"),
            "fromServiceWorker": req_meta.get("fromServiceWorker"),
        }

        # Save response body in transaction directory
        transaction_dir = req_meta.get("transactionDir")
        if transaction_dir:
            # Get proper extension for response body
            ext = self._guess_extension(req_meta.get("mimeType") or "") or ".bin"
            body_path = os.path.join(transaction_dir, f"response_body{ext}")
            try:
                n = self._write_body_file(body_path, data, is_b64)
                response_info.update({"bodyPath": body_path, "bodyBytes": n})
            except Exception as e:
                response_info["bodySaveError"] = str(e)
        else:
            response_info["bodySaveError"] = "No transaction directory available"

        # Response data already saved in transaction directory
        transaction_dir = req_meta.get("transactionDir")
        if transaction_dir:
            write_json_file(os.path.join(transaction_dir, "response.json"), response_info)

    def _on_fetch_get_body_reply(self, cmd_id, msg, cdp_session):
        ctx = self.fetch_get_body_wait.pop(cmd_id)
        rid = ctx["rid"]
        body_info = msg.get("result", {}) or {}
        body = body_info.get("body", "")
        is_b64 = bool(body_info.get("base64Encoded"))

        # Save to disk (best effort)
        try:
            with open(ctx["body_path"], mode="wb") as f:
                if is_b64:
                    f.write(base64.b64decode(body))
                else:
                    f.write((body or "").encode("utf-8", errors="replace"))
        except Exception:
            pass

        # Log + pair JSON - only response-specific data
        response_info = {
            "url": ctx.get("url"),
            "status": ctx.get("status"),
            "responseHeaders": ctx.get("headers", {}),
            "bodyPath": ctx.get("body_path"),
        }
        try:
            size = os.path.getsize(ctx["body_path"]) if os.path.exists(ctx["body_path"]) else None
            response_info["bodyBytes"] = size
        except Exception:
            pass

        # Response data already saved in transaction directory
        if ctx.get("transaction_dir"):
            # Save response metadata in transaction directory
            write_json_file(os.path.join(ctx["transaction_dir"], "response.json"), response_info)

        # IMPORTANT: unpause the response so the page receives the body
        self._safe_continue_response(self.session_registry.get(ctx.get("_sess_id"), cdp_session), rid)


    def get_network_summary(self):
        return {
            "requests_tracked": len(self.req_meta),
            "pending_bodies": len(self.pending_body)
        }

    def generate_har_file(self, output_path=None, page_title="Bluebox Session"):
        """
        Generate a HAR (HTTP Archive) file from collected network data.
        
        Args:
            output_path: Path to save the HAR file. If None, saves to output_dir/network.har
            page_title: Title for the HAR page entry
            
        Returns:
            dict: The HAR data structure
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "network.har")

        har_data = self._create_har_structure(page_title)

        # Convert network data to HAR entries
        entries = []
        for request_id, meta in self.req_meta.items():
            if meta.get("type") in self.capture_resources:
                entry = self._create_har_entry(request_id, meta)
                if entry:
                    entries.append(entry)

        har_data["log"]["entries"] = entries

        # Save HAR file
        try:
            with open(output_path, mode='w', encoding='utf-8') as f:
                json.dump(har_data, f, indent=2, ensure_ascii=False)
            logger.info("HAR file saved to: %s", output_path)
        except Exception as e:
            logger.info("Failed to save HAR file: %s", e)

        return har_data

    def _create_har_structure(self, page_title):
        """Create the basic HAR file structure."""
        return {
            "log": {
                "version": "1.2",
                "creator": {
                    "name": "Bluebox Network Monitor",
                    "version": "1.0"
                },
                "browser": {
                    "name": "Chrome DevTools Protocol",
                    "version": "1.0"
                },
                "pages": [
                    {
                        "startedDateTime": datetime.now().isoformat() + "Z",
                        "id": "page_1",
                        "title": page_title,
                        "pageTimings": {
                            "onContentLoad": -1,
                            "onLoad": -1
                        }
                    }
                ],
                "entries": []
            }
        }

    def _create_har_entry(self, request_id, meta):
        """Convert network request/response data to HAR entry format."""
        try:
            # Calculate timing (simplified - using timestamps if available)
            start_time = meta.get("ts", time.time() * 1000)
            duration = 100  # Default duration in ms
            
            # Create HAR entry
            entry = {
                "pageref": "page_1",
                "startedDateTime": datetime.fromtimestamp(start_time / 1000).isoformat() + "Z",
                "time": duration,
                "request": self._create_har_request(meta),
                "response": self._create_har_response(meta),
                "cache": {},
                "timings": {
                    "blocked": -1,
                    "dns": -1,
                    "connect": -1,
                    "send": 0,
                    "wait": duration,
                    "receive": 0
                },
                "connection": "0"
            }
            
            return entry
            
        except Exception as e:
            logger.info("Error creating HAR entry for request %s: %s", request_id, e)
            return None

    def _create_har_request(self, meta):
        """Create HAR request object from network metadata."""
        request_headers = meta.get("requestHeaders", {})
        headers = [{"name": k, "value": v} for k, v in request_headers.items()]
        
        # Parse URL for query string
        url = meta.get("url", "")
        query_string = []
        if "?" in url:
            url_part, query_part = url.split("?", 1)
            for param in query_part.split("&"):
                if "=" in param:
                    name, value = param.split("=", 1)
                    query_string.append({"name": name, "value": value})
        
        # Parse cookies from headers
        cookies = []
        cookie_header = request_headers.get("Cookie", "")
        if cookie_header:
            for cookie in cookie_header.split(";"):
                cookie = cookie.strip()
                if "=" in cookie:
                    name, value = cookie.split("=", 1)
                    cookies.append({"name": name, "value": value})
        
        post_data = None
        post_data_text = meta.get("postData")
        if post_data_text:
            post_data = {
                "mimeType": "application/x-www-form-urlencoded",
                "text": post_data_text
            }
        
        return {
            "method": meta.get("method", "GET"),
            "url": url,
            "httpVersion": "HTTP/1.1",
            "headers": headers,
            "queryString": query_string,
            "cookies": cookies,
            "headersSize": -1,
            "bodySize": len(post_data_text) if post_data_text else 0,
            "postData": post_data
        }

    def _create_har_response(self, meta):
        """Create HAR response object from network metadata."""
        response_headers = meta.get("responseHeaders", {})
        headers = [{"name": k, "value": v} for k, v in response_headers.items()]
        
        # Parse cookies from response headers
        cookies = []
        set_cookies = meta.get("setCookies", [])
        for cookie in set_cookies:
            cookies.append({
                "name": cookie.get("name", ""),
                "value": cookie.get("value", ""),
                "path": cookie.get("path", ""),
                "domain": cookie.get("domain", ""),
                "expires": cookie.get("expires", ""),
                "httpOnly": cookie.get("httpOnly", False),
                "secure": cookie.get("secure", False)
            })
        
        # Try to get response body content
        content_text = ""
        content_size = 0
        transaction_dir = meta.get("transactionDir")
        
        if transaction_dir:
            response_body_file = self._find_response_body_file(transaction_dir)
            if response_body_file:
                try:
                    file_size = os.path.getsize(response_body_file)
                    content_size = file_size
                    
                    # Try to read as text for HAR content
                    file_ext = os.path.splitext(response_body_file)[1].lower()
                    if file_ext in ['.json', '.xml', '.html', '.txt', '.js', '.css']:
                        with open(response_body_file, 'r', encoding='utf-8', errors='replace') as f:
                            content_text = f.read()
                except Exception:
                    pass
        
        return {
            "status": meta.get("status", 0),
            "statusText": meta.get("statusText", ""),
            "httpVersion": "HTTP/1.1",
            "headers": headers,
            "cookies": cookies,
            "content": {
                "size": content_size,
                "mimeType": meta.get("mimeType", ""),
                "text": content_text
            },
            "redirectURL": "",
            "headersSize": -1,
            "bodySize": content_size
        }

    # ------------------------ Helpers ------------------------
    def _safe_continue_response(self, cdp_session, rid):
        try:
            cdp_session.send("Fetch.continueResponse", {"requestId": rid})
        except Exception:
            pass

    def _safe_continue_request(self, cdp_session, rid):
        try:
            cdp_session.send("Fetch.continueRequest", {"requestId": rid})
        except Exception:
            pass

    @staticmethod
    def _guess_extension(mime_type: str) -> str:
        mt = (mime_type or "").lower().strip()

        # Common direct matches
        if "json" in mt: return ".json"
        if "html" in mt: return ".html"
        if "javascript" in mt or "ecmascript" in mt: return ".js"
        if "css" in mt: return ".css"
        if "xml" in mt: return ".xml"
        if "plain" in mt: return ".txt"
        if "csv" in mt: return ".csv"
        if "yaml" in mt or "yml" in mt: return ".yaml"

        # Images
        if "png" in mt: return ".png"
        if "jpeg" in mt or "jpg" in mt: return ".jpg"
        if "gif" in mt: return ".gif"
        if "bmp" in mt: return ".bmp"
        if "webp" in mt: return ".webp"
        if "tiff" in mt or "tif" in mt: return ".tiff"
        if "svg" in mt: return ".svg"
        if "ico" in mt: return ".ico"
        if "heic" in mt: return ".heic"
        if "avif" in mt: return ".avif"

        # PDFs / Documents
        if "pdf" in mt: return ".pdf"
        if "msword" in mt: return ".doc"
        if "vnd.openxmlformats-officedocument.wordprocessingml" in mt: return ".docx"
        if "vnd.ms-excel" in mt: return ".xls"
        if "vnd.openxmlformats-officedocument.spreadsheetml" in mt: return ".xlsx"
        if "vnd.ms-powerpoint" in mt: return ".ppt"
        if "vnd.openxmlformats-officedocument.presentationml" in mt: return ".pptx"
        if "rtf" in mt: return ".rtf"
        if "markdown" in mt or "md" in mt: return ".md"

        # Audio
        if "mpeg" in mt and "audio" in mt: return ".mp3"
        if "aac" in mt: return ".aac"
        if "wav" in mt: return ".wav"
        if "ogg" in mt: return ".ogg"
        if "flac" in mt: return ".flac"
        if "midi" in mt: return ".mid"
        if "webm" in mt and "audio" in mt: return ".weba"

        # Video
        if "mp4" in mt: return ".mp4"
        if "x-matroska" in mt or "matroska" in mt: return ".mkv"
        if "webm" in mt and "video" in mt: return ".webm"
        if "quicktime" in mt: return ".mov"
        if "avi" in mt: return ".avi"
        if "mpeg" in mt and "video" in mt: return ".mpeg"

        # Archives / Binary bundles
        if "zip" in mt: return ".zip"
        if "tar" in mt: return ".tar"
        if "gzip" in mt or "x-gzip" in mt: return ".gz"
        if "rar" in mt: return ".rar"
        if "7z" in mt: return ".7z"

        # Fonts
        if "font" in mt or "opentype" in mt: return ".otf"
        if "truetype" in mt: return ".ttf"
        if "woff2" in mt: return ".woff2"
        if "woff" in mt: return ".woff"

        # Other structured data
        if "protobuf" in mt: return ".proto"
        if "msgpack" in mt: return ".msgpack"
        if "bson" in mt: return ".bson"

        # Images disguised as octet-stream
        if "octet-stream" in mt:
            # Some common patterns inside the name
            if "exe" in mt: return ".exe"
            if "dll" in mt: return ".dll"
            if "bin" in mt: return ".bin"

        # Last resort
        return ".txt"


    def _should_block_url(self, url: str) -> bool:
        """Check if a URL should be blocked based on block_patterns."""
        if not self.block_patterns or not url:
            return False
            
        # Check against each block pattern (supports wildcards like Network.setBlockedURLs)
        for pattern in self.block_patterns:
            if fnmatch(url, pattern):
                return True
                
        return False

    def consolidate_transactions(self, output_file_path=None):
        """
        Consolidate all transactions into a single JSON file.
        
        Returns dict with structure:
        {
            "dir_name": {
                "request": {...},
                "response": {...}, 
                "response_body": "..." or preview
            }
        }
        """
        transactions_dir = self.paths.get("transactions_dir", os.path.join(self.output_dir, "transactions"))
        
        if not os.path.exists(transactions_dir):
            return {}
        
        consolidated = {}
        
        # Iterate through all transaction directories
        for dir_name in os.listdir(transactions_dir):
            dir_path = os.path.join(transactions_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
                
            transaction_data = {}
            
            # Read request.json
            request_file = os.path.join(dir_path, "request.json")
            if os.path.exists(request_file):
                try:
                    with open(request_file, 'r', encoding='utf-8') as f:
                        transaction_data["request"] = json.load(f)
                except Exception as e:
                    transaction_data["request"] = {"error": f"Failed to read request: {str(e)}"}
            
            # Read response.json
            response_file = os.path.join(dir_path, "response.json")
            if os.path.exists(response_file):
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        transaction_data["response"] = json.load(f)
                except Exception as e:
                    transaction_data["response"] = {"error": f"Failed to read response: {str(e)}"}
            
            # Read response body (with smart preview)
            response_body_file = self._find_response_body_file(dir_path)
            if response_body_file:
                transaction_data["response_body"] = self._get_response_body_content(response_body_file)
            else:
                transaction_data["response_body"] = None
                
            consolidated[dir_name] = transaction_data
        
        # Save to file if output path provided
        if output_file_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(consolidated, f, indent=2, ensure_ascii=False)
                logger.info("Consolidated transactions saved to: %s", output_file_path)
            except Exception as e:
                logger.error("Failed to save consolidated transactions: %s", e, exc_info=True)
        
        return consolidated

    def generate_har_from_transactions(self, output_path=None, page_title="Bluebox Session"):
        """
        Generate HAR file from transaction directories with full response bodies.
        This reads the full response body files directly instead of using truncated consolidated data.
        
        Args:
            output_path: Path to save the HAR file. If None, saves to output_dir/network.har
            page_title: Title for the HAR page entry
            
        Returns:
            dict: The HAR data structure
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "network.har")
        
        har_data = self._create_har_structure(page_title)
        
        # Read transaction directories directly instead of using consolidated data
        transactions_dir = self.paths.get("transactions_dir", os.path.join(self.output_dir, "transactions"))
        
        if not os.path.exists(transactions_dir):
            logger.info("Transactions directory not found: %s", transactions_dir)
            return har_data
        
        # Convert transaction directories to HAR entries
        entries = []
        transaction_dirs = [d for d in os.listdir(transactions_dir) if os.path.isdir(os.path.join(transactions_dir, d))]
        total_dirs = len(transaction_dirs)
        logger.info(f"Processing {total_dirs} transaction directories for HAR file...")
        
        for idx, dir_name in enumerate(transaction_dirs, 1):
            if idx % 50 == 0 or idx == total_dirs:
                logger.info(f"Processing HAR entry {idx}/{total_dirs}...")
            dir_path = os.path.join(transactions_dir, dir_name)
            entry = self._create_har_entry_from_directory(dir_name, dir_path)
            if entry:
                entries.append(entry)
        
        har_data["log"]["entries"] = entries
        
        # Save HAR file
        if output_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(har_data, f, indent=2, ensure_ascii=False)
                logger.info("HAR file saved to: %s", output_path)
            except Exception as e:
                logger.error("Failed to save HAR file: %s", e, exc_info=True)
        
        return har_data

    def _create_har_entry_from_directory(self, dir_name, dir_path):
        """Create HAR entry by reading files directly from transaction directory."""
        try:
            # Read request.json
            request_file = os.path.join(dir_path, "request.json")
            if not os.path.exists(request_file):
                return None
                
            with open(request_file, mode='r', encoding='utf-8') as f:
                request_data = json.load(f)
            
            # Read response.json
            response_file = os.path.join(dir_path, "response.json")
            response_data = {}
            if os.path.exists(response_file):
                with open(response_file, mode='r', encoding='utf-8') as f:
                    response_data = json.load(f)
            
            # Read full response body
            response_body_file = self._find_response_body_file(dir_path)
            full_response_body = ""
            response_body_size = 0
            
            if response_body_file:
                try:
                    response_body_size = os.path.getsize(response_body_file)
                    file_ext = os.path.splitext(response_body_file)[1].lower()
                    if file_ext in ['.json', '.xml', '.html', '.txt', '.js', '.css']:
                        with open(response_body_file, mode='r', encoding='utf-8', errors='replace') as f:
                            full_response_body = f.read()
                    else:
                        full_response_body = f"[Binary content - {response_body_size} bytes]"
                except Exception as e:
                    full_response_body = f"[Error reading file: {str(e)}]"
            
            # Calculate timing (simplified)
            duration = 100  # Default duration in ms
            
            # Create HAR entry
            entry = {
                "pageref": "page_1",
                "startedDateTime": datetime.now().isoformat() + "Z",
                "time": duration,
                "request": self._create_har_request_from_transaction(request_data),
                "response": self._create_har_response_from_directory(response_data, full_response_body, response_body_size),
                "cache": {},
                "timings": {
                    "blocked": -1,
                    "dns": -1,
                    "connect": -1,
                    "send": 0,
                    "wait": duration,
                    "receive": 0
                },
                "connection": "0"
            }
            
            return entry
            
        except Exception as e:
            logger.info("Error creating HAR entry from directory %s: %s", dir_name, e)
            return None

    def _create_har_entry_from_transaction(self, dir_name, transaction_data):
        """Create HAR entry from transaction directory data."""
        try:
            request_data = transaction_data.get("request", {})
            response_data = transaction_data.get("response", {})
            
            if not request_data:
                return None
            
            # Calculate timing (simplified)
            duration = 100  # Default duration in ms
            
            # Create HAR entry
            entry = {
                "pageref": "page_1",
                "startedDateTime": datetime.now().isoformat() + "Z",
                "time": duration,
                "request": self._create_har_request_from_transaction(request_data),
                "response": self._create_har_response_from_transaction(response_data, transaction_data.get("response_body"), request_data),
                "cache": {},
                "timings": {
                    "blocked": -1,
                    "dns": -1,
                    "connect": -1,
                    "send": 0,
                    "wait": duration,
                    "receive": 0
                },
                "connection": "0"
            }
            
            return entry
            
        except Exception as e:
            logger.info("Error creating HAR entry from transaction %s: %s", dir_name, e)
            return None

    def _create_har_request_from_transaction(self, request_data):
        """Create HAR request object from transaction request data."""
        request_headers = request_data.get("requestHeaders", {})
        headers = [{"name": k, "value": v} for k, v in request_headers.items()]
        
        # Parse URL for query string
        url = request_data.get("url", "")
        query_string = []
        if "?" in url:
            url_part, query_part = url.split("?", 1)
            for param in query_part.split("&"):
                if "=" in param:
                    name, value = param.split("=", 1)
                    query_string.append({"name": name, "value": value})
        
        # Parse cookies from headers
        cookies = []
        cookie_header = request_headers.get("Cookie", "")
        if cookie_header:
            for cookie in cookie_header.split(";"):
                cookie = cookie.strip()
                if "=" in cookie:
                    name, value = cookie.split("=", 1)
                    cookies.append({"name": name, "value": value})
        
        post_data = None
        post_data_text = request_data.get("postData")
        if post_data_text:
            post_data = {
                "mimeType": "application/x-www-form-urlencoded",
                "text": post_data_text
            }
        
        return {
            "method": request_data.get("method", "GET"),
            "url": url,
            "httpVersion": "HTTP/1.1",
            "headers": headers,
            "queryString": query_string,
            "cookies": cookies,
            "headersSize": -1,
            "bodySize": len(post_data_text) if post_data_text else 0,
            "postData": post_data
        }

    def _create_har_response_from_transaction(
        self, 
        response_data: dict[str, Any], 
        response_body_data: dict[str, Any] | str | None, 
        request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create HAR response object from transaction response data."""
        response_headers = response_data.get("responseHeaders", {})
        headers = [{"name": k, "value": v} for k, v in response_headers.items()]
        
        # Parse cookies from response headers
        cookies = []
        set_cookies = response_data.get("setCookies", [])
        for cookie in set_cookies:
            cookies.append({
                "name": cookie.get("name", ""),
                "value": cookie.get("value", ""),
                "path": cookie.get("path", ""),
                "domain": cookie.get("domain", ""),
                "expires": cookie.get("expires", ""),
                "httpOnly": cookie.get("httpOnly", False),
                "secure": cookie.get("secure", False)
            })

        # Get response body content - read full content from file, not truncated data
        content_text = ""
        content_size = 0

        # Try to read the full response body file directly
        transaction_dir = response_data.get("transactionDir")
        if not transaction_dir and request_data:
            transaction_dir = request_data.get("transactionDir")
        if transaction_dir:
            response_body_file = self._find_response_body_file(transaction_dir)
            if response_body_file:
                try:
                    file_size = os.path.getsize(response_body_file)
                    content_size = file_size

                    # Read full content for HAR file (no truncation)
                    file_ext = os.path.splitext(response_body_file)[1].lower()
                    if file_ext in ['.json', '.xml', '.html', '.txt', '.js', '.css']:
                        with open(response_body_file, mode='r', encoding='utf-8', errors='replace') as f:
                            content_text = f.read()
                    else:
                        # For binary files, include a note that content is binary
                        content_text = f"[Binary content - {file_size} bytes]"
                except Exception as e:
                    content_text = f"[Error reading file: {str(e)}]"
                    content_size = 0
        
        # Fallback to response_body_data if file reading fails
        if not content_text and response_body_data:
            if isinstance(response_body_data, dict):
                if response_body_data.get("type") == "full_content":
                    content_text = response_body_data.get("content", "")
                    content_size = response_body_data.get("size_bytes", 0)
                elif response_body_data.get("type") == "json_parsed":
                    content_text = json.dumps(response_body_data.get("content", {}))
                    content_size = response_body_data.get("size_bytes", 0)
                elif response_body_data.get("type") == "preview":
                    # For preview, try to get the full content from the file
                    content_text = response_body_data.get("preview", "")
                    content_size = response_body_data.get("size_bytes", 0)
            else:
                content_text = str(response_body_data)
                content_size = len(content_text)
        
        return {
            "status": response_data.get("status", 0),
            "statusText": response_data.get("statusText", ""),
            "httpVersion": "HTTP/1.1",
            "headers": headers,
            "cookies": cookies,
            "content": {
                "size": content_size,
                "mimeType": response_data.get("mimeType", ""),
                "text": content_text
            },
            "redirectURL": "",
            "headersSize": -1,
            "bodySize": content_size
        }

    def _create_har_response_from_directory(self, response_data, full_response_body, response_body_size):
        """Create HAR response object from directory files with full response body."""
        response_headers = response_data.get("responseHeaders", {})
        headers = [{"name": k, "value": v} for k, v in response_headers.items()]
        
        # Parse cookies from response headers
        cookies = []
        set_cookies = response_data.get("setCookies", [])
        for cookie in set_cookies:
            cookies.append({
                "name": cookie.get("name", ""),
                "value": cookie.get("value", ""),
                "path": cookie.get("path", ""),
                "domain": cookie.get("domain", ""),
                "expires": cookie.get("expires", ""),
                "httpOnly": cookie.get("httpOnly", False),
                "secure": cookie.get("secure", False)
            })
        
        return {
            "status": response_data.get("status", 0),
            "statusText": response_data.get("statusText", ""),
            "httpVersion": "HTTP/1.1",
            "headers": headers,
            "cookies": cookies,
            "content": {
                "size": response_body_size,
                "mimeType": response_data.get("mimeType", ""),
                "text": full_response_body
            },
            "redirectURL": "",
            "headersSize": -1,
            "bodySize": response_body_size
        }
    
    def _find_response_body_file(self, dir_path):
        """Find the response_body file (with any extension) in the directory."""
        for filename in os.listdir(dir_path):
            if filename.startswith("response_body"):
                return os.path.join(dir_path, filename)
        return None
    
    def _get_response_body_content(self, file_path):
        """Get response body content with smart preview based on file size (200 char limit)."""
        try:
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # For text files (JSON, XML, HTML, etc.)
            if file_ext in ['.json', '.xml', '.html', '.txt', '.js', '.css']:
                with open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    
                    # If content is 200 chars or less, return full content
                    if len(content) <= 200:
                        # For small JSON files, try to parse for structured data
                        if file_ext == '.json':
                            try:
                                parsed_content = json.loads(content)
                                return {
                                    "type": "json_parsed",
                                    "size_bytes": file_size,
                                    "content": parsed_content
                                }
                            except json.JSONDecodeError:
                                pass
                        
                        return {
                            "type": "full_content",
                            "size_bytes": file_size,
                            "content": content
                        }
                    
                    # If content is longer than 200 chars, return preview
                    else:
                        preview = content[:200]
                        return {
                            "type": "preview",
                            "size_bytes": file_size,
                            "preview": preview + "...",
                            "note": f"Content truncated (showing first 200 of {len(content)} characters)"
                        }
            
            # For binary files, return metadata only
            else:
                return {
                    "type": "binary",
                    "size_bytes": file_size,
                    "file_extension": file_ext,
                    "note": "Binary file - content not included"
                }
                
        except Exception as e:
            return {
                "type": "error",
                "error": str(e)
            }

    @staticmethod
    def _write_body_file(path: str, data: str, is_b64: bool) -> int:
        if is_b64:
            raw = base64.b64decode(data or b"")
            with open(path, mode="wb") as f:
                f.write(raw)
            return len(raw)
        else:
            with open(path, mode="w", encoding="utf-8", errors="replace") as f:
                f.write(data or "")
            return len(data or "")
