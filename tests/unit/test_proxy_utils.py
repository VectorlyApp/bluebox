"""Tests for bluebox/utils/proxy_utils.py"""

import pytest

from bluebox.utils.proxy_utils import parse_proxy_address


class TestParseProxyAddress:
    def test_host_port_only(self) -> None:
        result = parse_proxy_address("myproxy.com:8080")
        assert result.host_port == "myproxy.com:8080"
        assert result.username is None
        assert result.password is None
        assert not result.has_auth

    def test_http_scheme_no_auth(self) -> None:
        result = parse_proxy_address("http://myproxy.com:8080")
        assert result.host_port == "myproxy.com:8080"
        assert result.username is None
        assert result.password is None
        assert not result.has_auth

    def test_with_auth_no_scheme(self) -> None:
        result = parse_proxy_address("user:pass@myproxy.com:8080")
        assert result.host_port == "myproxy.com:8080"
        assert result.username == "user"
        assert result.password == "pass"
        assert result.has_auth

    def test_with_auth_http_scheme(self) -> None:
        result = parse_proxy_address("http://user:pass@myproxy.com:8080")
        assert result.host_port == "myproxy.com:8080"
        assert result.username == "user"
        assert result.password == "pass"
        assert result.has_auth

    def test_special_chars_in_password_percent_encoded(self) -> None:
        result = parse_proxy_address("http://user:p%40ss%3Aword@host:9090")
        assert result.host_port == "host:9090"
        assert result.username == "user"
        assert result.password == "p%40ss%3Aword"

    def test_whitespace_stripped(self) -> None:
        result = parse_proxy_address("  http://host:8080  ")
        assert result.host_port == "host:8080"
        assert not result.has_auth

    def test_ip_address_proxy(self) -> None:
        result = parse_proxy_address("216.98.230.152:6605")
        assert result.host_port == "216.98.230.152:6605"
        assert not result.has_auth

    def test_ip_address_with_auth(self) -> None:
        result = parse_proxy_address("http://myuser:mypass@216.98.230.152:6605")
        assert result.host_port == "216.98.230.152:6605"
        assert result.username == "myuser"
        assert result.password == "mypass"
        assert result.has_auth

    def test_username_only_no_password(self) -> None:
        """Username without password should not count as has_auth."""
        result = parse_proxy_address("http://user@host:8080")
        assert result.host_port == "host:8080"
        assert result.username == "user"
        assert result.password is None
        assert not result.has_auth
