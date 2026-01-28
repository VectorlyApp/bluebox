import warnings

warnings.warn(
    "web-hacker has been renamed to bluebox-sdk. "
    "Please update: pip install bluebox-sdk",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from bluebox for backward compatibility
from bluebox import *
