"""
clean_data.py  [Step 2: CLEAN]
==============================
Purpose   Make raw data consistent so every later step behaves predictably
Input     Raw DataFrame + target column name
Output    Clean DataFrame, guaranteed to contain the target
Called by main.py (training) and api.py (stage 3, same cleaning for live requests)
Next      validate.py

Why it matters
- Stateless fixes only: column names, duplicates, rows without a target, index reset
- Nothing that LEARNS from data (means, bins, scaling): that belongs in features.py, after the split
- Same input always gives the same output, so it is safe to rerun
- Report how many rows were dropped: a sudden jump signals an upstream problem

Tests     tests/test_clean_data.py (stage 2)
"""
