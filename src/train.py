"""
train.py  [Step 5: TRAIN]
=========================
Purpose   Bundle the feature recipe + the algorithm into ONE pipeline and fit it
Input     X_train, y_train + unfitted recipe from features.py + model settings
Output    One fitted pipeline (preprocessing + model), saved by main.py as models/model.joblib
Called by main.py          Next   evaluate.py

Why it matters
- .fit() runs on the train split only, so validation and test metrics stay honest
- One artifact for preprocessing + model: what you test is exactly what you deploy
- Model choice and hyperparameters come from config: swap models without rewriting code
- Fixed random seed: same data, same result

Tests     tests/test_train.py (stage 2)
"""
