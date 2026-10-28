# data/inference/  [New data to score]

- **Purpose:** new, unlabelled records the trained model has never seen
- **Read by:** `src/main.py`, then passed to `src/infer.py`
- **Rule:** same columns as training data, without the target
