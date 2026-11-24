# [Project name: to be decided]

## Setup

You need [Git](https://git-scm.com/) and [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html).
Run every command from the repository root.

1. **Clone the repository**

   ```bash
   git clone https://github.com/tito-canto/2026-IE-MLOps-Course-Group4.git
   cd 2026-IE-MLOps-Course-Group4
   ```

2. **Create and activate the environment** (Python 3.10, pinned packages, test and style tools)

   ```bash
   conda env create -f environment.yml
   conda activate insurance-mlops
   ```

   To update an existing environment after `environment.yml` changes, run
   `conda env update -f environment.yml --prune`.

   Without conda: use Python 3.11 or newer (some pins in `requirements.txt` need it), then
   `python -m venv .venv`, activate it, and run
   `pip install -r requirements.txt pytest-cov flake8`.

3. **Add the data.** Download `insurance.csv` from
   [Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)
   and save it as `data/raw/insurance.csv`. The `data/` folders are git-ignored, so the file
   is never committed. You can also keep it elsewhere and pass `--data` (see below).

4. **Secrets.** None are needed yet. `.env.example` lists the names that later stages will
   need; copy it to `.env` and fill in values only when a stage asks for them. Never commit `.env`.

5. **Check the setup.** Both commands should finish without errors:

   ```bash
   python -m pytest --cov=src --cov-branch
   python -m flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests
   ```

## Running the pipeline

```bash
python -m src.main                                     # uses data/raw/insurance.csv
python -m src.main --data "C:\path\to\insurance.csv"   # or any local copy
```

The command cleans and validates the data, trains the notebook's OLS model, and scores the
final model on the reserved sample (10% of rows, never used for fitting). It writes the
cleaned CSV, model bundle and reserved-sample predictions to the paths in `config.yaml`, then
applies the **quality gate** (`quality_gate` in `config.yaml`): if the reserved-sample MAE is
above the threshold, it logs `Quality gate failed` and exits with code 1. With the current
threshold (0.90 × the measured baseline MAE, see [Goal and success metrics](#goal-and-success-metrics))
the current OLS model **fails the gate**; this is the expected, honest result until a better
model is trained.

Predictions from this command are held-out evaluation predictions. For new records use
`python -m src.infer --data <new-records.csv>`. Each step can also run on its own:
`python -m src.load_data`, `src.clean_data`, `src.validate`, `src.train` and
`python -m src.evaluate --data <held-out.csv>`. Raw data and generated outputs remain untracked.

`src/features.py` defines the shared feature recipe. Training selects the categorical
reference levels on the training split, fits median imputation on the appropriate fit
data, and saves the fitted recipe with the OLS model. Inference also accepts older
bundles containing only the fitted imputer and frozen feature columns.

> The insurance company should invest in an MLOps cost-prediction solution to reduce operational costs and improve accuracy over manual historical comparisons.

## 1. Business case

### Context

- **Client:** Sad Insurers Associated, health insurance industry, analytics department
- **Maturity:**
  - Data: LOW, because the dataset contains approximately 1,300 rows only
  - Tools: LOW, because the client does not provide stronger hardware, so we are limited to personal devices
  - Processes: LOW, because we are currently building the workflow structure for delivery and responsibility assignment
  - People: MEDIUM, because the analytics department consists of a limited number of employees with experience in the area
- **Owner after delivery:** internal analytics team (us)

### Problem and baseline

The sales team wants to predict how much it will cost the company to provide coverage to a prospective client based on their personal context.

Today they determine this based on previous costs for customers with similar contexts.

**Baseline:** insurance cost = amount billed per customer, source: historical financial statements. Value: To be measured.

### Goal and success metrics

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Customer cost-underestimation rate (percentage of customers whose actual medical charges exceed the estimated coverage cost) | To be measured | At least 20% below the baseline rate within 6 months of deployment (proposed target) |
| ML metric (quality gate) | `mae` on unseen data | `mae` of the similar-customers method, To be measured | `model_mae <= 0.90 * baseline_mae` (proposed target) |
| Limit | Response time | n/a | Under 1 second |
| Limit | Cost per prediction | n/a | Under $0.01 |

- The business KPI uses a relative improvement because the current underestimation rate is unknown
- Before deployment, both `mae` values are calculated on the same held-out dataset to ensure a fair comparison
- The 10% and 20% improvements are proposed targets, not measured company results

### Solution

The sales team receives an estimate of what will be billed yearly for the person, to determine available insurance plans for the customer (in the future it could also be used to classify whether to provide coverage or not).

The model predicts the amount billed from `age`, `sex`, `bmi`, `children`, `smoker` and `region`, on request.

If the model is unavailable or unsure: fall back to generic plan options.

### Data

- **Source:** open-source dataset obtained from Kaggle ([link](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)), 1,338 rows, 7 columns. One row represents one person who bought insurance
- **Target:** `charges`, the medical insurance cost billed to the individual, yearly
- **Availability:** input data will be captured on demand during meetings with the client
- **Refresh:** data will be refreshed using new historical data captured throughout the year
- **Data safety:** everything is encrypted, and devices format themselves after more than 10 wrong password attempts in a row

### Value, cost and risks

- **Benefit over a non-AI approach:** more accurate cost estimates than manually selecting customers with similar characteristics, reducing financial losses from underestimating coverage costs. Measured by a 20% reduction in the customer cost-underestimation rate
- **Rough cost:** 1 data scientist or ML engineer, 1 data engineer and 1 sales stakeholder; Python, scikit-learn and existing company infrastructure; approximately 6 to 8 weeks
- **Scales to:** other insurance products, geographical regions, customer segments and claims-cost forecasting

| Risk | Mitigation |
| --- | --- |
| Biased predictions | Evaluate errors across customer segments |
| Inaccurate predictions | Monitor `mae` and flag high-risk estimates for human review |
| Poor generalisation | Validate on unseen data and monitor model performance |

### Open gaps

- **Project name:** not decided
- **Context:** realistic target level (manual, pipeline or automated delivery) not chosen
- **Problem:** pain point (what goes wrong with the current method) not stated; baseline value, unit and period To be measured
- **Metrics:** `mae` does not measure the direction of errors, so the model could pass the gate while underestimating more customers (accepted risk, no mitigation yet)
- **Solution:** legal or policy basis for using `sex` and `bmi` not documented; future coverage decisions in or out of scope not decided; person responsible for the final decision not named; no rule for what triggers the fallback; GitHub Pages cannot serve real-time predictions, so hosting must be revisited
- **Data:** owner, licence and whether the data is simulated To be confirmed; label source not stated; retraining frequency not set, and labels only arrive after a full year of coverage; self-reported `smoker` and `bmi` not addressed; personal data yes/no and minimisation not stated
- **Risks:** no data quality or operational risk listed; `mae` monitoring is delayed by about a year with no early-warning monitor; infrastructure cost conflicts with tools rated LOW and is not labelled as an estimate
- **Key takeaway:** client not named; no reproducibility, testing or monitoring benefit named; mentions operating costs, which the KPI does not measure
