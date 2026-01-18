# Fetch Returns HTML Instead of JSON

> Related: [fetch.md](../operations/fetch.md), [js-evaluate.md](../operations/js-evaluate.md)

**Symptom:** Response is HTML error page, not API data

**Causes:**
- Wrong URL (redirected to login page)
- Missing auth (see [unauthenticated.md](unauthenticated.md))
- CORS blocked

**Diagnose:** Use `js_evaluate` to inspect what you got:
```javascript
(function() {
  const raw = sessionStorage.getItem('my_fetch_result');
  console.log('Response type:', typeof raw);
  console.log('First 500 chars:', raw?.substring(0, 500));
  console.log('Looks like HTML:', raw?.includes('<html') || raw?.includes('<!DOCTYPE'));
  return raw;
})()
```

**Fix:** If you're getting HTML, parse it with JS to extract the data you need:
```javascript
(function() {
  const html = sessionStorage.getItem('my_fetch_result');
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');

  // Extract what you need from the HTML
  return {
    title: doc.querySelector('title')?.textContent,
    data: Array.from(doc.querySelectorAll('table tr')).map(row => ({
      cells: Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim())
    })),
    links: Array.from(doc.querySelectorAll('a')).map(a => ({
      text: a.textContent.trim(),
      href: a.href
    }))
  };
})()
```
