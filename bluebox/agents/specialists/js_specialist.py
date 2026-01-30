"""
bluebox/agents/specialists/js_specialist.py

JavaScript specialist.
"""

from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class JSSpecialist(AbstractSpecialist):
    """
    JavaScript specialist.
    """
