# Click/Input Fails - Element Not Found

> Related: [ui-operations.md](../operations/ui-operations.md), [data-extraction.md](../operations/data-extraction.md)

**Symptom:** "Element not found" or click has no effect

**Diagnose:** Use `return_html` to see actual DOM:
```json
{"type": "return_html"}
```

Or use `js_evaluate` to check:
```javascript
(function() {
  const el = document.querySelector('#my-button');
  console.log('Found:', !!el);
  console.log('Tag:', el?.tagName);
  console.log('Visible:', el?.offsetParent !== null);
  return el?.outerHTML;
})()
```

**Solutions:**

| Problem | Fix |
|---------|-----|
| Element not loaded yet | Add `sleep` before interaction |
| Wrong selector | Use browser DevTools to verify selector |
| Element in iframe | Not supported - use fetch instead |
| Dynamic ID | Use attribute selector: `[data-testid="submit"]` |
