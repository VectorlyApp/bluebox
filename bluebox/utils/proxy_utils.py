"""
bluebox/utils/proxy_utils.py

Proxy address parsing utilities.
"""

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class ProxyCredentials:
    """Parsed proxy address components."""
    host_port: str
    username: str | None
    password: str | None

    @property
    def has_auth(self) -> bool:
        """Whether proxy credentials are present."""
        return self.username is not None and self.password is not None


def parse_proxy_address(proxy_address: str) -> ProxyCredentials:
    """
    Parse a proxy address string into its components.

    Supported formats:
    - "host:port"                  -> ProxyCredentials("host:port", None, None)
    - "http://host:port"           -> ProxyCredentials("host:port", None, None)
    - "user:pass@host:port"        -> ProxyCredentials("host:port", "user", "pass")
    - "http://user:pass@host:port" -> ProxyCredentials("host:port", "user", "pass")

    Args:
        proxy_address: Raw proxy address string.

    Returns:
        ProxyCredentials with parsed components.
    """
    addr = proxy_address.strip()

    # Normalize: add scheme for urlparse to work correctly
    if not addr.startswith(("http://", "https://")):
        addr = f"http://{addr}"

    parsed = urlparse(addr)

    host = parsed.hostname or ""
    port = parsed.port
    host_port = f"{host}:{port}" if port else host

    username = parsed.username if parsed.username else None
    password = parsed.password if parsed.password else None

    return ProxyCredentials(
        host_port=host_port,
        username=username,
        password=password,
    )
