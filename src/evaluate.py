"""
evaluate.py  [Step 6: EVALUATE]
===============================
Purpose   Score the fitted pipeline on data it has never seen
Input     Fitted pipeline + X_eval, y_eval (validation split; test split once, at the end)
Output    Dictionary of plain numbers, e.g. {"pr_auc": 0.61, "roc_auc": 0.84}
Called by main.py          Feeds  stage 3: the quality gate and experiment tracking

Why it matters
- Never score on training data: it hides overfitting
- Metric choice follows the business goal and problem type (set in config)
- Plain numbers are easy to log, compare between runs and turn into a go/no-go decision

Tests     tests/test_evaluate.py (stage 2)
"""
