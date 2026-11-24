# tests/  [Automated tests, from stage 2]

- **Purpose:** prove each module does its one job, every time code changes
- **Convention:** one test file per module: `src/clean_data.py` -> `tests/test_clean_data.py`
- **Run:** `pytest` from the repo root. Stage 2 runs it automatically on every Pull Request
- **Integration:** the stage test files end with an `integration: <module> -> <next module>` section
  that feeds the module's real output into the next pipeline stage
  (utils -> load_data / infer, load_data -> clean_data, clean_data -> validate, validate -> train)
- **Shared data:** `conftest.py` generates an insurance-like dataset (the real one is personal
  data and is not in Git) plus a messy copy and a temporary project folder with a full config
- **Coverage:** `pytest --cov=src --cov-branch` (target: 95 % branch coverage)
