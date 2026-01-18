# Execution

> How routines are executed via Chrome DevTools Protocol (CDP).

**Code:** [routine.py](web_hacker/data_models/routine/routine.py) (`Routine.execute()`), [execution.py](web_hacker/data_models/routine/execution.py)

## Execution Flow

1. **Create/attach browser tab** - New incognito tab or attach to existing `tab_id`
2. **Enable CDP domains** - Page, Runtime, Network, DOM
3. **Execute operations sequentially** - Each operation interpolates parameters, resolves placeholders, executes via CDP
4. **Collect result** - Final data from `return` or `return_html` operation
5. **Cleanup** - Close tab (unless `close_tab_when_done=False`)

## RoutineExecutionResult

```python
class RoutineExecutionResult(BaseModel):
    ok: bool = True                           # Success/failure
    error: str | None = None                  # Error message if failed
    warnings: list[str] = []                  # Non-fatal warnings
    data: dict | list | str | None = None     # Final result data
    operations_metadata: list[OperationExecutionMetadata] = []  # Per-operation timing/details
    placeholder_resolution: dict[str, str | None] = {}  # Resolved placeholder values
    is_base64: bool = False                   # True if data is base64-encoded binary
    content_type: str | None = None           # MIME type (for downloads)
    filename: str | None = None               # Suggested filename (for downloads)
```

## RoutineExecutionContext

A **mutable context** passed to each operation. Operations read from it and write results back:

```python
class RoutineExecutionContext(BaseModel):
    # CDP connection
    session_id: str                    # CDP session ID
    ws: WebSocket                      # WebSocket connection
    send_cmd: Callable                 # CDP command sender
    recv_until: Callable               # CDP response receiver

    # Input
    parameters_dict: dict = {}         # User-provided parameters
    timeout: float = 180.0             # Operation timeout

    # Mutable state (operations update these)
    current_url: str = "about:blank"   # Updated by navigate operations
    result: RoutineExecutionResult     # Final result - operations set result.data
    current_operation_metadata: OperationExecutionMetadata | None  # Current op metadata
```

**How operations mutate context:**
- `navigate` → updates `current_url`
- `fetch` → stores response in sessionStorage, updates `result.placeholder_resolution`
- `return` / `return_html` → sets `result.data`
- `download` → sets `result.data`, `result.is_base64`, `result.filename`
- All operations → append to `result.operations_metadata`

## Operation Metadata

Every operation automatically records execution metadata:

```python
class OperationExecutionMetadata(BaseModel):
    type: str              # Operation type (e.g., "fetch", "click")
    duration_seconds: float  # Execution time
    details: dict = {}     # Operation-specific data
    error: str | None      # Error if operation failed
```

**What gets stored in `details`:**
- `fetch` → `request`, `response` (method, url, status, headers)
- `click` → `selector`, `element` (tag, id, classes), `click_coordinates`
- `input_text` → `selector`, `text_length`, `element`
- `js_evaluate` → `console_logs`, `execution_error`, `storage_error`

Access after execution:
```python
result = routine.execute(params)
for op_meta in result.operations_metadata:
    print(f"{op_meta.type}: {op_meta.duration_seconds:.2f}s")
    if op_meta.details.get("response"):
        print(f"  Status: {op_meta.details['response']['status']}")
```

## Operation Execution

Each operation's `execute()` method:

1. Creates `OperationExecutionMetadata` with `type`
2. Calls `_execute_operation()` (operation-specific logic)
3. Operation mutates `context.result` and adds to `details`
4. Records `duration_seconds` and any `error`
5. Appends metadata to `context.result.operations_metadata`

```python
# Simplified from RoutineOperation.execute()
def execute(self, context):
    context.current_operation_metadata = OperationExecutionMetadata(type=self.type)
    start = time.perf_counter()
    try:
        self._execute_operation(context)  # Subclass implements this
    except Exception as e:
        context.current_operation_metadata.error = str(e)
    finally:
        context.current_operation_metadata.duration_seconds = time.perf_counter() - start
        context.result.operations_metadata.append(context.current_operation_metadata)
```

## Error Handling

- **CDP errors** - Connection/protocol failures → `result.ok = False`, `result.error` set
- **Operation errors** - JS/fetch failures → Raised as `RuntimeError`, caught at routine level
- **Placeholder warnings** - Unresolved placeholders → Added to `result.warnings`

## Download Results

For `download` operations, result contains base64 data:

```python
if result.is_base64 and result.filename:
    import base64
    with open(result.filename, "wb") as f:
        f.write(base64.b64decode(result.data))
```
