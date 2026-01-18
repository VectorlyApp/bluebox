# Placeholder Not Resolved

> Related: [placeholders.md](../core/placeholders.md), [fetch.md](../operations/fetch.md)

**Symptom:** Literal `{{param}}` appears in request, or value is empty

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| String not escape-quoted | Use `"\"{{param}}\""` not `"{{param}}"` |
| Storage placeholder in navigate | Not supported - only user params work in URLs |
| Storage placeholder in js_evaluate | Access directly: `sessionStorage.getItem('key')` |
| Wrong path | Check exact key name and nesting |

**Check what's in storage:**
```javascript
(function() {
  return JSON.parse(sessionStorage.getItem('my_key'));
})()
```
