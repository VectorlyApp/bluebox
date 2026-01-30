"""
bluebox/constants/network.py

Shared constants for network filtering, API detection, and content type handling.
"""

import re


AUTH_HEADERS: frozenset[str] = frozenset([
    "authorization",
    "x-api-key",
    "x-auth-token",
    "x-access-token",
    "api-key",
    "bearer",
])

EXCLUDED_MIME_PREFIXES: tuple[str, ...] = (
    "application/javascript",
    "application/x-javascript",
    "text/javascript",
    "image/",
    "video/",
    "audio/",
    "font/",
    "application/font",
    "application/octet-stream",
)

INCLUDED_MIME_PREFIXES: tuple[str, ...] = (
    "application/json",
    "text/html",
    "text/xml",
    "application/xml",
    "text/plain",
)

API_KEY_TERMS: tuple[str, ...] = (
    # Core API identifiers
    "api",
    "graphql",
    "rest",
    "rpc",
    # Authentication & User
    "auth",
    "login",
    "logout",
    "oauth",
    "token",
    "session",
    "user",
    "account",
    "profile",
    "register",
    "signup",
    # Data Operations
    "search",
    "query",
    "filter",
    "fetch",
    "create",
    "update",
    "delete",
    "submit",
    "save",
    # E-commerce/Transactions
    "cart",
    "checkout",
    "order",
    "payment",
    "purchase",
    "booking",
    "reserve",
    "quote",
    "price",
    # Content
    "content",
    "data",
    "item",
    "product",
    "result",
    "detail",
    "info",
    "summary",
    "list",
    # Actions
    "action",
    "execute",
    "process",
    "validate",
    "verify",
    "confirm",
    "send",
    # Events & Tracking
    "event",
    "track",
    "analytic",
    "metric",
    "log",
    # Autocomplete & Suggestions
    "autocomplete",
    "typeahead",
    "suggest",
    "complete",
    "hint",
    "predict",
    # Backend Services
    "gateway",
    "service",
    "backend",
    "internal",
    "ajax",
    "xhr",
    "bff",
    # Next.js / frameworks
    "_next/data",
    "__api__",
    "_api",
)

API_VERSION_PATTERN: re.Pattern[str] = re.compile(r"/v\d+/", re.IGNORECASE)

THIRD_PARTY_TRACKING_ANALYTICS_DOMAINS: tuple[str, ...] = (
    # Analytics & Performance Monitoring
    "google-analytics.com",
    "googletagmanager.com",
    "analytics.google.com",
    "go-mpulse.net",
    "akamai.net",
    "newrelic.com",
    "nr-data.net",
    "segment.io",
    "segment.com",
    "mixpanel.com",
    "amplitude.com",
    "heap.io",
    "heapanalytics.com",
    "fullstory.com",
    "hotjar.com",
    "mouseflow.com",
    "clarity.ms",
    "matomo.",
    "piwik.",
    "posthog.com",
    "optimizely.com",
    "scorecardresearch.com",
    "quantserve.com",
    "krxd.net",
    "adobedtm.com",
    # Consent & Privacy
    "onetrust.com",
    "cookielaw.org",
    "trustarc.com",
    "cookiebot.com",
    "privacy-center.",
    "consentmanager.",
    # Advertising & Marketing
    "doubleclick.net",
    "googlesyndication.com",
    "googleadservices.com",
    "facebook.net",
    "facebook.com/tr",
    "fbcdn.net",
    "ads-twitter.com",
    "twitter.com/i/",
    "linkedin.com/li/",
    "adsrvr.org",
    "criteo.com",
    "criteo.net",
    "taboola.com",
    "outbrain.com",
    "adnxs.com",
    "rubiconproject.com",
    "pubmatic.com",
    "openx.net",
    "casalemedia.com",
    "demdex.net",
    "omtrdc.net",
    "2o7.net",
    "snapchat.com",
    "sc-static.net",
    # Social Widgets
    "platform.twitter.com",
    "connect.facebook.net",
    "platform.linkedin.com",
    # CDNs (static assets)
    "fonts.gstatic.com",
    "fonts.googleapis.com",
    "gstatic.com",
    "jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "cloudflare.com/cdn-cgi/",
)

SKIP_FILE_EXTENSIONS: tuple[str, ...] = (
    ".js",
    ".css",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".mp4",
    ".webm",
    ".mp3",
    ".wav",
)
