"""Reproducible analysis for the supplied telecom churn dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Customer Churn.csv"
OUTPUT_DIR = ROOT / "output"
RANDOM_STATE = 42
TENURE_BANDS = ["0-12", "13-24", "25-48", "49+"]

CODED_CATEGORICAL_COLUMNS = [
    "complains",
    "age_group",
    "tariff_plan",
    "age",
]


def load_data(path=DATA_PATH):
    """Load the supplied CSV, normalize column names, and validate the target."""
    data = pd.read_csv(path)
    data.columns = (
        data.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )
    if "churn" not in data:
        raise ValueError("The CSV must contain a 'Churn' target column.")
    if data["churn"].isna().any():
        raise ValueError("The churn target contains missing values.")
    if not set(data["churn"].unique()).issubset({0, 1}):
        raise ValueError("Expected churn labels encoded as 0 and 1.")
    return data


def engineer_features(data):
    """Create a broad tenure band while retaining the original continuous value."""
    features = data.drop(columns="churn").copy()
    if "subscription_length" in features:
        features["subscription_length_band"] = pd.cut(
            features["subscription_length"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=TENURE_BANDS,
        ).astype("object")
    return features


def _build_preprocessor(features):
    categorical = [
        column
        for column in CODED_CATEGORICAL_COLUMNS + ["status", "subscription_length_band"]
        if column in features.columns
    ]
    numeric = [column for column in features.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def _capture_at_fraction(y_true, probabilities, fraction=0.30):
    count = max(1, int(np.ceil(len(probabilities) * fraction)))
    selected = np.argsort(probabilities)[-count:]
    total_churners = int(np.sum(y_true))
    captured = int(np.sum(np.asarray(y_true)[selected]))
    return {
        "target_fraction": fraction,
        "customers_selected": count,
        "churners_captured": captured,
        "churn_capture_rate": captured / total_churners if total_churners else 0.0,
        "precision_in_target_group": captured / count,
    }


def train_and_evaluate(data):
    """Fit two models and evaluate once on the same stratified holdout."""
    y = data["churn"].astype(int)
    features = engineer_features(data)

    # Status is excluded from the primary model until its point-in-time meaning
    # is verified; the sensitivity run shows the impact of including it.
    primary_features = features.drop(columns=["status"], errors="ignore")
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model_definitions = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    fitted_models = {}
    metric_rows = []
    predictions = {}

    for model_name, estimator in model_definitions.items():
        pipeline = Pipeline(
            [
                ("preprocessor", _build_preprocessor(primary_features)),
                ("model", estimator),
            ]
        )
        pipeline.fit(primary_features.iloc[train_idx], y.iloc[train_idx])
        probabilities = pipeline.predict_proba(primary_features.iloc[test_idx])[:, 1]
        predicted_labels = (probabilities >= 0.5).astype(int)
        top_group = _capture_at_fraction(y.iloc[test_idx].to_numpy(), probabilities)
        metric_rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(y.iloc[test_idx], probabilities),
                "pr_auc": average_precision_score(y.iloc[test_idx], probabilities),
                "precision_at_0_5": precision_score(
                    y.iloc[test_idx], predicted_labels, zero_division=0
                ),
                "recall_at_0_5": recall_score(
                    y.iloc[test_idx], predicted_labels, zero_division=0
                ),
                "top_30_percent_capture": top_group["churn_capture_rate"],
                "top_30_percent_precision": top_group["precision_in_target_group"],
                "top_30_percent_customers": top_group["customers_selected"],
                "top_30_percent_churners_captured": top_group["churners_captured"],
            }
        )
        fitted_models[model_name] = pipeline
        predictions[model_name] = probabilities

    # Sensitivity check for a field whose definition/timing is not documented.
    status_metrics = {}
    if "status" in features:
        status_preprocessor = _build_preprocessor(features)
        status_model = Pipeline(
            [
                ("preprocessor", status_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        status_model.fit(features.iloc[train_idx], y.iloc[train_idx])
        status_probabilities = status_model.predict_proba(features.iloc[test_idx])[:, 1]
        status_metrics = {
            "random_forest_roc_auc_with_status": roc_auc_score(
                y.iloc[test_idx], status_probabilities
            ),
            "random_forest_top_30_percent_capture_with_status": _capture_at_fraction(
                y.iloc[test_idx].to_numpy(), status_probabilities
            )["churn_capture_rate"],
        }

    return {
        "features": primary_features,
        "y": y,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "models": fitted_models,
        "predictions": predictions,
        "metrics": pd.DataFrame(metric_rows),
        "status_sensitivity": status_metrics,
    }


def _save_dashboard(data, results):
    OUTPUT_DIR.mkdir(exist_ok=True)
    churn_rate = data["churn"].mean()
    metrics = results["metrics"].set_index("model")
    rf = results["models"]["Random Forest"]
    x_test = results["features"].iloc[results["test_idx"]]
    y_test = results["y"].iloc[results["test_idx"]]
    rf_probabilities = results["predictions"]["Random Forest"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Telecom Customer Churn | Exploratory Dashboard", fontsize=17)
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.05,
        0.7,
        f"Customers\n{len(data):,}",
        fontsize=18,
        weight="bold",
        va="center",
    )
    axes[0, 0].text(
        0.52,
        0.7,
        f"Observed churn\n{churn_rate:.1%}",
        fontsize=18,
        weight="bold",
        va="center",
    )
    axes[0, 0].text(
        0.05,
        0.25,
        "Status excluded from the primary model pending\nconfirmation of its definition and timing.",
        fontsize=10,
        va="center",
    )

    segment_columns = [
        ("tariff_plan", "Tariff plan code"),
        ("complains", "Complaint code"),
        ("age_group", "Age-group code"),
    ]
    plot_axis = axes[0, 1]
    segment_labels = []
    segment_values = []
    for column, label in segment_columns:
        if column in data:
            grouped = data.groupby(column, observed=True)["churn"].mean()
            segment_labels.extend(f"{label}={code}" for code in grouped.index)
            segment_values.extend(grouped.to_list())
    plot_axis.bar(np.arange(len(segment_values)), segment_values, color="#70AD47")
    plot_axis.set_title("Churn rate by coded segment")
    plot_axis.set_ylabel("Observed churn rate")
    plot_axis.set_xlabel("Source code (meaning not provided)")
    plot_axis.set_xticks(np.arange(len(segment_labels)))
    plot_axis.set_xticklabels(segment_labels, rotation=45, ha="right", fontsize=8)

    for model_name, probabilities in results["predictions"].items():
        metric = metrics.loc[model_name]
        axes[1, 0].bar(
            model_name,
            metric["roc_auc"],
            label=f"ROC-AUC {metric['roc_auc']:.3f}",
        )
        axes[1, 0].text(
            model_name,
            metric["roc_auc"] - 0.04,
            f"Top-30% capture\n{metric['top_30_percent_capture']:.1%}",
            ha="center",
            color="white",
            fontsize=9,
        )
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("ROC-AUC")
    axes[1, 0].set_title("Holdout model results")
    axes[1, 0].tick_params(axis="x", labelrotation=10)

    try:
        importance = permutation_importance(
            rf,
            x_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=-1,
        )
        importance_table = pd.Series(
            importance.importances_mean, index=x_test.columns
        ).sort_values().tail(8)
        importance_table.plot(kind="barh", ax=axes[1, 1], color="#4472C4")
        axes[1, 1].set_title("Random Forest permutation importance")
        axes[1, 1].set_xlabel("Mean ROC-AUC decrease")
    except ValueError as exc:
        axes[1, 1].text(0.05, 0.5, f"Importance unavailable: {exc}")
        axes[1, 1].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "churn_dashboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ranked = np.argsort(-rf_probabilities)
    ranked_churn = y_test.to_numpy()[ranked]
    cumulative = np.cumsum(ranked_churn) / max(1, ranked_churn.sum())
    population_share = np.arange(1, len(ranked_churn) + 1) / len(ranked_churn)
    ax.plot(population_share, cumulative, label="Random Forest", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random targeting")
    ax.axvline(0.30, color="tomato", linestyle=":", label="Top 30%")
    ax.set(
        title="Cumulative churn capture on the holdout",
        xlabel="Share of customers ranked highest risk",
        ylabel="Share of holdout churners captured",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cumulative_gains.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_analysis():
    data = load_data()
    results = train_and_evaluate(data)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results["metrics"].to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    _save_dashboard(data, results)
    return data, results


if __name__ == "__main__":
    dataset, evaluation = run_analysis()
    print(f"Dataset: {len(dataset):,} rows, {dataset.shape[1]} columns")
    print(f"Observed churn rate: {dataset['churn'].mean():.1%}")
    print("\nHoldout metrics (status excluded):")
    print(evaluation["metrics"].round(3).to_string(index=False))
    if evaluation["status_sensitivity"]:
        print("\nStatus-field sensitivity (verify field timing/definition):")
        for metric, value in evaluation["status_sensitivity"].items():
            print(f"{metric}: {value:.3f}")
    print(f"\nFigures and metrics saved to: {OUTPUT_DIR}")
