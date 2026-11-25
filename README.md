# Telecom Customer Churn Analysis

An end-to-end churn-risk ranking project built from the `Customer Churn.csv` included in this folder. It compares an interpretable Logistic Regression baseline with a class-weighted Random Forest, evaluates churn ranking at a limited call capacity, and translates model lift into an explicitly assumption-based campaign scenario.

> **Dataset note:** The supplied CSV is not the IBM Telco Customer Churn dataset. It contains 3,150 rows and 14 columns, including `Subscription Length`, usage measures, numeric category codes, `Customer Value`, and a binary `Churn` target. It has no `TotalCharges`, contract type, monthly charges, payment method, internet service, add-on fields, or customer ID. Do not apply IBM Telco-specific conclusions to this file.

## Project files

- `Customer Churn.csv` — supplied source dataset.
- `churn_analysis.ipynb` — guided notebook with data checks, segment analysis, model evaluation, and campaign assumptions.
- `churn_analysis.py` — reusable analysis and visualization pipeline used by the notebook.
- `business_summary.md` — one-page summary of measured dataset facts and next steps.
- `output/churn_dashboard.png` — generated dashboard-style summary.
- `output/cumulative_gains.png` — holdout cumulative churn-capture curve.
- `output/model_metrics.csv` — holdout metrics for the two primary models.

## Run it

Use Python 3.10+ and install the dependencies if needed:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib jupyter
```

Run the end-to-end analysis:

```powershell
python churn_analysis.py
```

Or open `churn_analysis.ipynb` in Jupyter/VS Code and run all cells. The CSV must remain beside the notebook and script. Generated charts and metrics are written to `output/`.

## What the analysis does

1. Normalizes column names; checks the binary target, row count, missing values, duplicate rows, and churn rate.
2. Adds subscription-length bands (`0–12`, `13–24`, `25–48`, `49+`) while retaining the continuous feature. Confirm the unit with the data owner.
3. Uses a stratified 80/20 train/test split with a fixed seed and preprocessing pipelines for numeric and coded categorical fields.
4. Compares Logistic Regression and Random Forest using ROC-AUC, average precision (PR-AUC), precision/recall at a 0.5 threshold, and the fraction of test-set churners captured in the highest-risk 30%.
5. Excludes `Status` from primary models until its meaning and availability timing are verified, and reports a sensitivity check that includes it.
6. Writes dashboard figures and metrics to `output/`.

## Current observed results

On the supplied file, a quick profile found 3,150 customers, no missing values, and 495 churn labels (15.7%). Re-run `python churn_analysis.py` for the reproducible seed-42 holdout metrics recorded in `output/model_metrics.csv`. In the initial run, Random Forest ROC-AUC was 0.987 and its highest-risk 30% contained all 99 churners in the 630-row test split. This is a preliminary result from one random split—not an expected future campaign outcome. The codebook, temporal validation, data leakage risks, and campaign lift still need investigation.

## Interpretation and caveats

- Numeric code meanings were not provided. Rates by `Tariff Plan`, `Age Group`, `Complains`, `Age`, and `Status` are shown by code, not translated into unsupported labels.
- There is no customer ID, so a real retention call list cannot be delivered from this dataset alone.
- There is no monthly charge field. The notebook's financial example is a scenario controlled by explicit assumptions, not measured retained revenue.
- Churn prediction identifies association, not whether an offer prevents churn. Measure incremental retention with a randomized control group before scaling offers.
- Validate the date/time of every feature. In particular, `Status` may be a post-outcome or otherwise unavailable field; the primary model leaves it out pending clarification.
- A single holdout is not a substitute for out-of-time validation, subgroup checks, calibration, or ongoing monitoring.

## Next steps

1. Obtain the data dictionary, source timestamp, and customer key; verify that features are known before intervention.
2. Add dated snapshots and a later evaluation window to validate stability.
3. Run a randomized retention pilot and calculate incremental saved revenue, offer cost, and ROI.
4. Replace scenario assumptions with validated campaign economics and approved customer-contact capacity.
