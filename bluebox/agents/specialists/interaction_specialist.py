"""

bluebox/agents/specialists/interaction_specialist.py

Interaction specialist.
"""

from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist
from bluebox.llms.infra.interactions_data_store import InteractionsDataStore
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class InteractionSpecialist(AbstractSpecialist):
    """
    Interaction specialist.
    """

    pass