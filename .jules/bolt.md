## 2024-05-23 - [DPTHead Positional Embedding Optimization]
**Learning:** Precomputing values to avoid function calls can backfire if the overhead of managing the cache (loops, dictionary lookups) in Python exceeds the cost of the function calls, especially if those calls are already efficient. However, in this case, removing trigonometric calculations from the inner loop yielded a modest 2-3% speedup on CPU. The key is that "optimizations" must be measured.
**Action:** Always profile both "before" and "after". Don't assume caching is always faster.
