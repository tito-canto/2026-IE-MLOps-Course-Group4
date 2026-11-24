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

**Baseline (similar-customers method), measured:** each customer's cost is estimated as the
mean `charges` of past customers with the same smoker status, age band (under 30, 30–39,
40–49, 50+) and BMI band (WHO: under 18.5, 18.5–25, 25–30, 30+). Past customers are the
development rows; the estimate is scored on the reserved sample (134 customers, seed 47),
the same rows the model's quality gate uses.

| Measure (reserved sample, n = 134) | Similar-customers baseline | Current OLS model |
| --- | --- | --- |
| `mae` (in `charges` units per customer per year) | **2,423.5** | 4,035.4 |
| Cost-underestimation rate | **26.1%** | 41.0% |
| For reference: `mae` of predicting the overall mean | 9,932.0 | n/a |

The current OLS model does **not** beat the baseline: its MAE is 66% higher and it
underestimates more customers. Every group in the baseline had matching past customers, so
no fallback was needed. These values come from the historical dataset; the live rate must be
re-measured on real sales quotes.

### Goal and success metrics

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Customer cost-underestimation rate (percentage of customers whose actual medical charges exceed the estimated coverage cost) | 26.1% (historical data) | At least 20% below the baseline rate, i.e. 20.9% or lower, within 6 months of deployment (proposed target) |
| ML metric (quality gate) | `mae` on the reserved sample | 2,423.5 (similar-customers method) | `model_mae <= 0.90 * baseline_mae` = **2,181.2**, enforced as `quality_gate` in `config.yaml` |
| Limit | Response time | n/a | Under 1 second |
| Limit | Cost per prediction | n/a | Under $0.01 |

- Both `mae` values are calculated on the same reserved sample, so the comparison is fair
- Every `python -m src.main` run checks the gate and stops with `Quality gate failed` when the
  model misses it, so a run always says whether the model meets the MAE target
- The 10% and 20% improvements are proposed targets, not measured company results

#### Training loss vs. evaluation metric

The model is fitted with ordinary least squares (OLS), which minimises **squared** error and
predicts the average cost. The case is judged on **MAE**, the average absolute error in
`charges` units per customer, because that is the money the sales team gets wrong on a typical
quote. The two do not reward the same thing: squared error gives large errors far more weight,
so a few very expensive customers (mostly smokers with high BMI) pull the OLS fit, and a model
with a good squared error can still have a poor MAE.

We keep OLS for now because it is the notebook's agreed, explainable model, and we close the
gap by **judging every run on MAE** through the quality gate, not on the loss the model was
trained with. The current OLS fit fails that gate. Next options, in order: add the smoker ×
BMI and smoker × age interactions the baseline groups capture, then try a model that
minimises absolute error directly (median/quantile regression). Any replacement must pass the
same gate.

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

#### Personal data

**Yes, this is personal data, and part of it is health data.** In production every input row
describes a real, identifiable prospect captured during a sales meeting. `bmi` and `smoker`
reveal health information, which the GDPR treats as a special category (Art. 9): it needs an
explicit legal basis (for example explicit consent), a data protection impact assessment
before go-live, and only the fields the model really needs. The Kaggle training data has no
names or IDs, but it holds the same health fields.

Which fields we really need, based on the OLS fit on all development rows:

| Field | Sensitivity | Effect on `charges` (coefficient, p-value) | Decision |
| --- | --- | --- | --- |
| `smoker` | Health data | +23,450 for smokers, p < 0.001 (largest effect) | Needed |
| `age` | Personal | +261 per year, p < 0.001 | Needed |
| `bmi` | Health data | +338 per BMI point, p < 0.001 | Needed |
| `children` | Personal | +436 per child, p = 0.003 | Needed (small effect, low sensitivity) |
| `region` | Personal | Only `southeast` differs (−1,049, p = 0.039) | Under review; drop if a model without it passes the gate |
| `sex` | Personal, protected characteristic | −17, p = 0.96 (no effect) | **Not needed.** Remove from `config.yaml` in the next model iteration |

What we do **not** collect or keep: names, contact details, ID or policy numbers, and any
other health data. The model needs none of them. Inputs are used to produce a quote and are
not stored next to an identifier. The saved model drops its training rows (`remove_data()` in
`src/train.py`), and raw data and outputs stay out of Git (`.gitignore`). Device encryption
protects the data at rest, but minimisation is the main safeguard: the less we collect, the
less can leak.

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
- **Problem:** pain point (what goes wrong with the current method) not stated; the baseline is measured on historical data only and is computed by a one-off analysis, not yet a pipeline step
- **Model:** the current OLS model fails the MAE quality gate (4,035.4 vs. 2,181.2 required) and does not beat the similar-customers baseline; it is not fit for use
- **Metrics:** `mae` does not measure the direction of errors, so the model could pass the gate while underestimating more customers (accepted risk, no mitigation yet)
- **Solution:** legal or policy basis for using `sex` and `bmi` not documented; future coverage decisions in or out of scope not decided; person responsible for the final decision not named; no rule for what triggers the fallback; GitHub Pages cannot serve real-time predictions, so hosting must be revisited
- **Data:** owner, licence and whether the data is simulated To be confirmed; label source not stated; retraining frequency not set, and labels only arrive after a full year of coverage; self-reported `smoker` and `bmi` not addressed; legal basis for processing health data and the impact assessment not yet done; `sex` still in the model although it is not needed
- **Risks:** no data quality or operational risk listed; `mae` monitoring is delayed by about a year with no early-warning monitor; infrastructure cost conflicts with tools rated LOW and is not labelled as an estimate
- **Key takeaway:** client not named; no reproducibility, testing or monitoring benefit named; mentions operating costs, which the KPI does not measure
