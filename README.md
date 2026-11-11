# [Project name: to be decided]

## Running the pipeline

Create and activate the environment in `environment.yml`. Put the insurance CSV at
`data/raw/insurance.csv`, or supply its local path without copying it into the repository.
Replace the example path below with the location of your CSV:

```powershell
python -m src.main --data "C:\path\to\insurance.csv"
```

The command cleans and validates the data, trains the notebook's OLS model, and evaluates
the final model on the reserved sample. It writes the cleaned CSV, model bundle and
reserved-sample predictions to the configured paths. Predictions from this command are
held-out evaluation predictions; use `python -m src.infer --data <new-records.csv>`
for new records. The proposed business quality gate cannot be assessed until the
similar-customers baseline MAE is measured. Raw data and generated outputs remain untracked.

`src/features.py` defines the shared feature recipe. Training selects the categorical
reference levels on the training split, fits median imputation on the appropriate fit
data, and saves the fitted recipe with the OLS model. Inference also accepts older
bundles containing only the fitted imputer and frozen feature columns.

Run `python -m pytest --cov=src --cov-branch` for tests and coverage. Run
`python -m flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests`
for the course style check.

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
