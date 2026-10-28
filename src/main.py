"""
main.py  [Step 0: ORCHESTRATOR]
================================
Purpose   Run the whole pipeline, in order, with one command: python -m src.main
Owns      The order of steps, the train/validation/test split, passing data between steps
Never     Cleaning, feature or model logic. main.py only CALLS other modules

Flow
                 settings (stage 2: SETTINGS dict here | stage 3: config.yaml)
                                      |
    data/raw --> load_data --> clean_data --> validate --> SPLIT (train / val / test)
                                                                   |
                           features (unfitted recipe) --> train (fit on train only)
                                                                   |
                                              evaluate (validation split)
                                                                   |
                                         save --> models/model.joblib
                                                                   |
    data/inference --> infer --> reports/predictions.csv

Why it matters
- Split BEFORE anything is fitted, so no information leaks from test data into training
- Data passes as explicit variables between stateless steps: no hidden state
- One entry point that you, your teammates, CI and deployment all run the same way

Tests     tests/test_main.py (stage 2)
"""
