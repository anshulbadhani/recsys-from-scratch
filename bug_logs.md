# Bug logs
This file include bug encountered while the development. Does not contain trivial issues like: `unordered_map<int, string>` instead of `unordered_map<string, int>`. *This bug report was written by Claude while debugging*.

### Bug 1 – Deduplication after filtering in `03_filter_data.py`
**File:** `scripts/03_filter_data.py`
**Symptom:** Test — `ground truth item never appears in user train history` — FAIL. Same item appearing in both train and test for some users.
**Cause:** Users who reviewed the same item twice at the same timestamp had that item land in both splits. Deduplication was placed after filtering, meaning interaction counts were computed on duplicate rows.
```python
# wrong — dedup after filter, counts are inflated
df = df[df['user_id'].isin(user_counts >= 5)]
df = df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='last')

# correct — dedup first, then count
df = df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='last')
df = df[df['user_id'].isin(user_counts >= 5)]
```
**Fix:** Move `drop_duplicates` to step 0, before any filtering or sorting.