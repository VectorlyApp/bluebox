# AgentCard Implementation Plan

## What
Add `AgentCard` dataclass and `AGENT_CARD` class variable to `AbstractAgent`, enforced via `__init_subclass__`.

## Changes

### 1. `bluebox/agents/abstract_agent.py`
- Add `AgentCard` frozen dataclass with fields: `name: str`, `description: str`, `data_requirements: tuple[str, ...] = ()`
- Add `ClassVar` import
- Add `AGENT_CARD: ClassVar[AgentCard]` declaration on `AbstractAgent`
- Add `__init_subclass__` that validates non-abstract subclasses define `AGENT_CARD`
  - Skip condition: `cls.__name__.startswith("Abstract")` (consistent with existing pattern in AbstractSpecialist)
  - Chains via `super().__init_subclass__(**kwargs)` — AbstractSpecialist's existing `__init_subclass__` already calls super, so both will fire for specialists

### 2. Six concrete subclasses — add `AGENT_CARD` class variable:
- `bluebox/agents/super_discovery_agent.py` — SuperDiscoveryAgent
- `bluebox/agents/bluebox_agent.py` — BlueBoxAgent
- `bluebox/agents/specialists/network_specialist.py` — NetworkSpecialist
- `bluebox/agents/specialists/js_specialist.py` — JSSpecialist
- `bluebox/agents/specialists/interaction_specialist.py` — InteractionSpecialist
- `bluebox/agents/specialists/value_trace_resolver_specialist.py` — ValueTraceResolverSpecialist

### 3. Export `AgentCard` from `__init__.py` if one exists for agents (check).

## Design Notes
- `data_requirements` uses canonical loader names (e.g. `"network_data_loader"`) not constructor param names
- `description` is the 1-2 sentence summary an orchestrator would inject into its prompt
- Naming convention skip (`startswith("Abstract")`) matches existing pattern in AbstractSpecialist
