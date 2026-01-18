# Page Not Fully Loaded

> Related: [navigation.md](../operations/navigation.md), [ui-operations.md](../operations/ui-operations.md)

**Symptom:** Element not found, fetch returns unexpected data, click fails

**Solution:** Add more sleep time after navigation

```json
{"type": "navigate", "url": "https://example.com", "sleep_after_navigation_seconds": 5.0}
```

Or add explicit sleep:
```json
{"type": "sleep", "timeout_seconds": 3.0}
```
