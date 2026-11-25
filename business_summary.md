# Customer Churn Prediction — Business Summary

## Decision to support

Prioritize a limited set of customers for retention outreach using churn-risk ranking, then test whether the offer actually prevents churn.

## Data and observed facts

The supplied file contains 3,150 customer records and 14 columns. The target has 495 churn labels (15.7%); no missing values were observed in the initial profile. Available inputs include subscription length, usage measures, customer value, and numeric-coded fields. This is **not** the IBM Telco dataset in the original project description: contract type, monthly charges, payment method, internet service, add-ons, and customer ID are absent.

## Initial model result

Using a stratified 80/20 random split (seed 42), a class-weighted Random Forest excluding the undocumented `Status` feature achieved ROC-AUC **0.987**. Its top 30% risk group captured **99 of 99 churners** in the 630-customer test set. This is a single holdout result and requires temporal validation; it should not be treated as expected campaign performance. See `output/model_metrics.csv` after running the analysis for the full metrics table.

## Business interpretation

The observed holdout ranking suggests that a capacity-constrained pilot may be worth testing. However, the file has no unique customer identifier, no monthly revenue field, and no offer-response data. A revenue or ROI result therefore cannot be measured from this dataset. The notebook includes an illustrative scenario only, with visible assumptions.

## Recommended actions

1. Obtain a data dictionary and confirm when `Status`, `Age`, and coded fields are known relative to the retention decision.
2. Add stable customer IDs and dated feature snapshots so a model can generate an auditable contact list.
3. Validate ranking on a later time period and monitor performance across customer segments.
4. Randomize eligible high-risk customers between an offer and a control group; measure incremental retention and net revenue before expanding.
5. Do not claim contract, fiber, payment-method, autopay, or add-on drivers from this CSV; those fields are not present.

## Illustrative financial scenario

The analysis notebook provides editable assumptions for monthly revenue, offer success, offer cost, and target share. Any resulting retained revenue, campaign cost, or net benefit is a scenario—not a measured forecast—and depends on holdout lift carrying forward and offers causing incremental retention.
