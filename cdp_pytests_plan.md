# Test Plan: CDP Monitors, FileEventWriter, and AsyncCDPSession

## Overview

Create pytest test files for the CDP monitoring infrastructure:
- 5 monitor classes in `web_hacker/cdp/monitors/`
- `FileEventWriter` in `web_hacker/cdp/file_event_writer.py`
- `AsyncCDPSession` in `web_hacker/cdp/async_cdp_session.py`

## Test Strategy

### Tier 1: Pure/Static Methods (No Mocking Required)
These are the easiest to test and provide good coverage with minimal effort.

### Tier 2: File-Based Operations (tmpdir fixtures)
Methods that read/write files - use pytest's `tmp_path` fixture.

### Tier 3: CDP-Dependent Methods (Mock AsyncCDPSession)
Methods requiring CDP interaction - mock the session and verify behavior.

---

## Test Files to Create

### 1. `tests/unit/cdp/test_abstract_async_monitor.py`

**Tests:**
- `test_subclass_registration` - Verify all 4 monitors are registered in `_subclasses`
- `test_get_all_subclasses_returns_copy` - Ensure it returns a copy, not the original
- `test_get_monitor_category` - Each subclass returns its class name

### 2. `tests/unit/cdp/test_async_network_monitor.py`

**Tier 1 - Static methods:**
- `test_is_internal_url` - chrome:// returns True, https:// returns False, None returns False
- `test_is_static_asset` - .css/.png/etc return True, /api/ returns False
- `test_should_block_url` - googletagmanager blocked, normal URLs not blocked
- `test_get_set_cookie_values` - Extract Set-Cookie from various header formats
- `test_parse_json_if_applicable` - JSON content-type parses, others return string
- `test_is_html` - Detects HTML by content-type and body patterns
- `test_clean_response_body` - Truncation, JSON handling, HTML cleaning

**Tier 1 - Class method:**
- `test_get_ws_event_summary` - Returns correct summary structure with truncated URL

**Tier 2 - File operations:**
- `test_consolidate_transactions` - Reads JSONL, returns dict
- `test_consolidate_transactions_missing_file` - Returns empty dict
- `test_consolidate_transactions_with_output` - Writes to output file
- `test_generate_har_from_transactions` - Generates valid HAR structure
- `test_create_har_entry_from_event` - Creates proper HAR entry

### 3. `tests/unit/cdp/test_async_storage_monitor.py`

**Tier 1 - Class method:**
- `test_get_ws_event_summary_cookie_change` - Cookie change summary
- `test_get_ws_event_summary_storage_event` - Storage event summary

**Tier 3 - State management (mock callback):**
- `test_handle_dom_storage_added` - Updates local_storage_state
- `test_handle_dom_storage_updated` - Updates state and emits event
- `test_handle_dom_storage_removed` - Removes from state
- `test_handle_dom_storage_cleared` - Clears origin state
- `test_handle_get_cookies_reply_initial` - Sets initial cookie state
- `test_handle_get_cookies_reply_changes` - Detects added/modified/removed cookies

### 4. `tests/unit/cdp/test_async_window_property_monitor.py`

**Tier 1 - Static method:**
- `test_is_application_object_native_classname` - HTMLElement, SVGElement return False
- `test_is_application_object_native_name` - document, window return False
- `test_is_application_object_app_object` - Custom names with Object className return True

**Tier 1 - Class method:**
- `test_get_ws_event_summary` - Returns correct structure with change_count, changed_paths

### 5. `tests/unit/cdp/test_async_interaction_monitor.py`

**Tier 1 - Class method:**
- `test_get_ws_event_summary` - Returns interaction_type and element_tag

**Tier 1 - Instance method:**
- `test_parse_interaction_event` - Parses raw JS data into UiInteractionEvent
- `test_parse_interaction_event_missing_element` - Returns None for missing element

**Tier 2 - File operations:**
- `test_consolidate_interactions` - Reads JSONL and returns consolidated dict
- `test_consolidate_interactions_missing_file` - Returns empty structure

### 6. `tests/unit/cdp/test_file_event_writer.py`

**Tier 2 - File operations:**
- `test_init_creates_directories` - Parent directories created
- `test_write_event_network` - Writes to network events file
- `test_write_event_storage` - Writes to storage events file
- `test_write_event_window_properties` - Writes to window properties file
- `test_write_event_interaction` - Writes to interaction events file
- `test_write_event_unknown_category` - Logs warning, doesn't write
- `test_write_event_pydantic_model` - Calls model_dump() if available
- `test_create_from_output_dir` - Factory creates correct paths structure

### 7. `tests/unit/cdp/test_async_cdp_session.py`

**Tier 3 - Mocked WebSocket/CDP:**
- `test_init_creates_monitors` - All 4 monitors instantiated
- `test_send_increments_seq` - Sequence ID increments
- `test_send_includes_session_id` - SessionId added when available
- `test_enable_domain_idempotent` - Second call skips enable
- `test_handle_message_routes_to_network` - Network events go to network monitor
- `test_handle_message_routes_to_storage` - Storage events go to storage monitor
- `test_handle_command_reply` - Resolves pending future
- `test_get_monitoring_summary` - Aggregates all monitor summaries

---

## File Structure

Following existing project convention (tests in `tests/unit/` directly):

```
tests/unit/
├── test_cdp_monitors.py         # All 5 monitor classes in one file
├── test_file_event_writer.py    # FileEventWriter tests
└── test_async_cdp_session.py    # AsyncCDPSession tests

tests/conftest.py                # Add CDP fixtures to existing conftest
```

## Shared Fixtures (add to `tests/conftest.py`)

```python
@pytest.fixture
def mock_event_callback():
    """Async callback that records calls for CDP monitor testing."""
    from unittest.mock import AsyncMock
    return AsyncMock()

@pytest.fixture
def mock_cdp_session():
    """Mock AsyncCDPSession for testing monitors."""
    from unittest.mock import AsyncMock
    session = AsyncMock()
    session.send = AsyncMock(return_value=1)
    session.send_and_wait = AsyncMock(return_value={"result": {}})
    session.enable_domain = AsyncMock()
    session.page_session_id = "mock-session-id"
    return session
```

## Implementation Order

1. **tests/conftest.py** - Add CDP fixtures to existing conftest
2. **tests/unit/test_cdp_monitors.py** - All monitor tests (AbstractAsyncMonitor, AsyncNetworkMonitor, AsyncStorageMonitor, AsyncWindowPropertyMonitor, AsyncInteractionMonitor)
3. **tests/unit/test_file_event_writer.py** - FileEventWriter tests
4. **tests/unit/test_async_cdp_session.py** - AsyncCDPSession tests

## Verification

After implementation:
```bash
# Run all new tests
pytest tests/unit/test_cdp_monitors.py tests/unit/test_file_event_writer.py tests/unit/test_async_cdp_session.py -v

# Run with coverage to verify we're testing the right code
pytest tests/unit/test_cdp_monitors.py tests/unit/test_file_event_writer.py tests/unit/test_async_cdp_session.py -v --cov=web_hacker.cdp.monitors --cov=web_hacker.cdp.file_event_writer --cov=web_hacker.cdp.async_cdp_session --cov-report=term-missing
```

## Notes

- Please see other existing tests to get an idea for how we like to organize our pytests.
- **Skip heavy CDP mocking initially** - Focus on Tier 1 and Tier 2 tests first for quick wins
- **Async tests use `@pytest.mark.asyncio`** - Requires `pytest-asyncio` (already in deps)
- **File tests use `tmp_path` fixture** - Built into pytest, no setup needed
