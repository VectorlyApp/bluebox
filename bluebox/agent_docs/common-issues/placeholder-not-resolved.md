# Placeholder Not Resolved

> Placeholders like `{{param}}` appear as literal text or resolve to empty values due to wrong parameter type, using storage placeholders in navigate (not supported), wrong storage access in js_evaluate, undefined parameters, or incorrect paths. Related: [placeholders.md](../core/placeholders.md), [fetch.md](../operations/fetch.md)

**Symptom:** Literal `{{param}}` appears in request, or value is empty

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Parameter not defined | Add the parameter to the `parameters` list in the routine |
| Wrong parameter type | Match the type to what the raw CDP request sends (e.g., if `"5"` is a string, use `type=string`) |
| Storage placeholder in navigate | Not supported - only user params work in URLs |
| Storage placeholder in js_evaluate | Access directly: `sessionStorage.getItem('key')` |
| Wrong path | Check exact key name and nesting |

**Check what's in storage:**
```javascript
(function() {
  return JSON.parse(sessionStorage.getItem('my_key'));
})()
```
