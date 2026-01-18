# Agent Routine Execution Timeout

**Symptom:** Routine execution times out or fails after extended run time

**Possible Causes:**

1. **Long-running routine** - Routine takes 3+ minutes to complete
2. **Network traffic issues** - Connectivity problems with the Chrome deployment server
3. **Bad JS evaluation** - JavaScript code stuck in infinite loop or taking too long to execute

**Solutions:**

1. **Run again** - Transient network issues often resolve on retry
2. **Optimize routine** - Break long routines into smaller parts, reduce unnecessary sleeps
3. **Check JS code** - Ensure js_evaluate operations don't have infinite loops or expensive computations
4. **Verify network** - Check connectivity to the Chrome deployment server

**Still stuck?** Consider reaching out to the Vectorly support team for assistance.
