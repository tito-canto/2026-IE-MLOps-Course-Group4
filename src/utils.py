"""
utils.py  [Shared plumbing]
===========================
Purpose   Small, reusable read and write helpers
Examples  read and write CSV files, save and load the model file
Used by   load_data.py, main.py, infer.py, api.py (stage 3)

Why it matters
- One place for file I/O (Input/Output): fix a bug once, every module benefits
- No business logic: if a helper knows your target column, it belongs elsewhere

Tests     tests/test_utils.py (stage 2)
"""
