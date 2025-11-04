"""
src/data_models/network.py

Network data models.
"""

from enum import StrEnum


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

class Method(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    TRACE = "TRACE"
    CONNECT = "CONNECT"
    
class Stage(StrEnum):
    REQUEST = "Request"
    RESPONSE = "Response"
