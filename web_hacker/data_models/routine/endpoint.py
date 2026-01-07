"""
web_hacker/data_models/routine/endpoint.py

Parameter models with comprehensive validation and type information.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResourceType(StrEnum):
    XHR = "XHR"
    FETCH = "Fetch"
    SCRIPT = "Script"
    DOCUMENT = "Document"
    IMAGE = "Image"
    STYLESHEET = "Stylesheet"
    FONT = "Font"
    MEDIA = "Media"
    OTHER = "Other"


class RESPONSE_TYPES(StrEnum):
    """
    Supported response types for API endpoints.
    """
    ARRAYBUFFER = "arraybuffer"
    JSON = "json"
    TEXT = "text"
    BLOB = "blob"
    FORMDATA = "formdata"


class HTTPMethod(StrEnum):
    """
    Supported HTTP methods for API endpoints.
    """
    # standard
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    # informational
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"
    # tunnel/diagnostic
    CONNECT = "CONNECT"
    TRACE = "TRACE"


class CREDENTIALS(StrEnum):
    """
    Supported credentials modes for API requests.
    """
    SAME_ORIGIN = "same-origin"
    INCLUDE = "include"
    OMIT = "omit"


class MimeType(StrEnum):
    """
    Common MIME types for file downloads and content type detection.
    Reference: https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    """
    # Default/Binary
    OCTET_STREAM = "application/octet-stream"
    
    # Documents
    PDF = "application/pdf"
    JSON = "application/json"
    XML = "application/xml"
    ZIP = "application/zip"
    GZIP = "application/gzip"
    
    # Microsoft Office
    DOC = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    XLS = "application/vnd.ms-excel"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    PPT = "application/vnd.ms-powerpoint"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    
    # OpenDocument
    ODT = "application/vnd.oasis.opendocument.text"
    ODS = "application/vnd.oasis.opendocument.spreadsheet"
    ODP = "application/vnd.oasis.opendocument.presentation"
    
    # Text
    PLAIN = "text/plain"
    HTML = "text/html"
    CSS = "text/css"
    CSV = "text/csv"
    JAVASCRIPT = "text/javascript"
    MARKDOWN = "text/markdown"
    
    # Images
    PNG = "image/png"
    JPEG = "image/jpeg"
    GIF = "image/gif"
    WEBP = "image/webp"
    SVG = "image/svg+xml"
    BMP = "image/bmp"
    ICO = "image/vnd.microsoft.icon"
    TIFF = "image/tiff"
    AVIF = "image/avif"
    APNG = "image/apng"
    
    # Audio
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    OGG_AUDIO = "audio/ogg"
    AAC = "audio/aac"
    WEBM_AUDIO = "audio/webm"
    MIDI = "audio/midi"
    
    # Video
    MP4 = "video/mp4"
    WEBM_VIDEO = "video/webm"
    OGG_VIDEO = "video/ogg"
    MPEG = "video/mpeg"
    AVI = "video/x-msvideo"
    
    # Fonts
    WOFF = "font/woff"
    WOFF2 = "font/woff2"
    TTF = "font/ttf"
    OTF = "font/otf"


class Endpoint(BaseModel):
    """
    Endpoint model with comprehensive parameter validation.
    """
    url: str = Field(
        ...,
        description="Target API URL with parameter placeholders"
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the fetch request"
    )
    method: HTTPMethod = Field(
        ...,
        description="HTTP method"
    )
    headers: dict[str, Any] | None = Field(
        default=None,
        description="Dictionary of headers, with parameter placeholders for later interpolation"
    )
    body: dict[str, Any] | None = Field(
        default=None,
        description="Dictionary of request body, with parameter placeholders for later interpolation"
    )
    credentials: CREDENTIALS = Field(
        default=CREDENTIALS.SAME_ORIGIN,
        description="Credentials mode, defaults to 'same-origin'"
    )
