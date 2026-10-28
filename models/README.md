# models/  [Trained model, generated]

- **Purpose:** holds the saved model, e.g. `model.joblib`
- **Written by:** `src/main.py` after the model passes evaluation (through `utils.py`)
- **Read by:** `src/infer.py`, and in stage 3 the API (`src/api.py`)
- **Key idea:** ONE file with preprocessing + model together, so training and serving cannot drift apart
- **Git:** ignored. Stage 3 versions models in a model registry instead
