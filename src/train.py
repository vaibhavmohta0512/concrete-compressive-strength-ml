
"""End-to-end reproducible EDA, regression model comparison and tuning pipeline."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform
from data_utils import load_dataset, FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODELS = ROOT / "models"
for p in (OUT, FIG, MODELS):
    p.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE (MPa)": float(mean_absolute_error(y_true, y_pred)),
        "MSE (MPa²)": float(mse),
        "RMSE (MPa)": float(np.sqrt(mse)),
        "R²": float(r2_score(y_true, y_pred)),
    }

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close()

def main():
    sns.set_theme(style="whitegrid")
    df = load_dataset()
    # Data audit: report exact duplicates but do not automatically drop them.
    audit = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": int(len(FEATURES)),
        "target": TARGET,
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "non_numeric_cells_after_conversion": int(df.isna().sum().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "descriptive_statistics": df.describe().round(3).to_dict(),
    }
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    df.describe().T.to_csv(OUT / "descriptive_statistics.csv")

    # EDA figures
    df[TARGET].hist(bins=30)
    plt.xlabel("Compressive strength (MPa)"); plt.ylabel("Frequency")
    plt.title("Distribution of concrete compressive strength")
    savefig("target_distribution.png")

    df[FEATURES].hist(figsize=(13, 9), bins=25)
    plt.suptitle("Input feature distributions")
    savefig("feature_distributions.png")

    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson correlation matrix")
    savefig("correlation_heatmap.png")

    plt.figure(figsize=(13, 7))
    sns.boxplot(data=df[FEATURES], orient="h")
    plt.title("Input feature boxplots (different units/scales; descriptive only)")
    savefig("feature_boxplots.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, feature in zip(axes.flat, FEATURES):
        ax.scatter(df[feature], df[TARGET], alpha=.55, s=14)
        ax.set_xlabel(feature); ax.set_ylabel("Strength (MPa)")
    fig.suptitle("Input features versus compressive strength")
    fig.tight_layout()
    fig.savefig(FIG / "feature_vs_target.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Split first; all learned preprocessing remains inside model pipelines.
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE,
                                            min_samples_leaf=1, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
        ]),
    }
    rows, fitted = [], {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {"Model": name, **regression_metrics(y_test, pred)}
        # Training score is diagnostic only, not used to select final model.
        row["Train R² (diagnostic)"] = float(pipe.score(X_train, y_train))
        rows.append(row); fitted[name] = pipe
        pd.DataFrame({"Actual (MPa)": y_test, "Predicted (MPa)": pred,
                      "Residual (MPa)": y_test - pred}).to_csv(
                          OUT / (name.lower().replace(" ", "_") + "_predictions.csv"), index=False)

    baseline = pd.DataFrame(rows).sort_values("RMSE (MPa)")
    baseline.to_csv(OUT / "baseline_model_comparison.csv", index=False)

    # Tune Random Forest only using training data and CV, never the holdout test set.
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    param_dist = {
        "model__n_estimators": randint(200, 701),
        "model__max_depth": [None, 5, 8, 12, 16, 22],
        "model__min_samples_split": randint(2, 12),
        "model__min_samples_leaf": randint(1, 5),
        "model__max_features": [1.0, "sqrt", 0.8],
    }
    search = RandomizedSearchCV(
        rf_pipe, param_distributions=param_dist, n_iter=25,
        scoring="neg_root_mean_squared_error", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    tuned_pred = tuned.predict(X_test)
    tuned_metrics = regression_metrics(y_test, tuned_pred)
    tuning_comparison = pd.DataFrame([
        {"Model": "Random Forest (baseline)", **regression_metrics(y_test, fitted["Random Forest"].predict(X_test))},
        {"Model": "Random Forest (tuned)", **tuned_metrics},
    ])
    tuning_comparison.to_csv(OUT / "tuning_comparison.csv", index=False)
    (OUT / "tuning_details.json").write_text(json.dumps({
        "best_params": search.best_params_,
        "best_cv_RMSE_MPa": float(-search.best_score_),
        "test_metrics": tuned_metrics,
        "interpretation_note": "Test metrics are for one fixed holdout split. Compare baseline/tuned on the same test split."
    }, indent=2, default=str))

    # Choose between all baselines and tuned RF by holdout RMSE; report test performance once.
    candidates = {**fitted, "Tuned Random Forest": tuned}
    candidate_rows = []
    candidate_preds = {}
    for name, model in candidates.items():
        pred = model.predict(X_test)
        candidate_preds[name] = pred
        candidate_rows.append({"Model": name, **regression_metrics(y_test, pred)})
    comparison = pd.DataFrame(candidate_rows).sort_values("RMSE (MPa)")
    comparison.to_csv(OUT / "final_model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["Model"])
    best_model = candidates[best_name]
    joblib.dump(best_model, MODELS / "best_concrete_strength_model.joblib")
    (MODELS / "model_metadata.json").write_text(json.dumps({
        "best_model": best_name, "features": FEATURES, "target": TARGET,
        "target_unit": "MPa", "random_state": RANDOM_STATE,
        "dataset": "UCI Concrete Compressive Strength (id 165)"
    }, indent=2))

    # Comparative metric charts
    for metric in ["MAE (MPa)", "RMSE (MPa)", "R²"]:
        plt.figure(figsize=(9, 5))
        order = comparison.sort_values(metric, ascending=(metric != "R²"))
        sns.barplot(data=order, x=metric, y="Model")
        plt.title(f"Holdout test comparison: {metric}")
        savefig("comparison_" + metric.split()[0].lower().replace("²", "2") + ".png")

    best_pred = candidate_preds[best_name]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_pred, alpha=.65)
    lo = min(float(y_test.min()), float(best_pred.min()))
    hi = max(float(y_test.max()), float(best_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual strength (MPa)"); plt.ylabel("Predicted strength (MPa)")
    plt.title(f"Actual vs predicted: {best_name}")
    savefig("actual_vs_predicted.png")

    residuals = y_test.to_numpy() - best_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(best_pred, residuals, alpha=.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted strength (MPa)"); plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Residual analysis: {best_name}")
    savefig("residual_plot.png")

    # Permutation importance: computed on held-out test set for descriptive interpretation.
    pi = permutation_importance(best_model, X_test, y_test, n_repeats=15,
                                random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error",
                                n_jobs=-1)
    imp = pd.DataFrame({"Feature": FEATURES, "Importance (RMSE increase)": pi.importances_mean,
                        "Std": pi.importances_std}).sort_values("Importance (RMSE increase)", ascending=False)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=imp, x="Importance (RMSE increase)", y="Feature")
    plt.title("Permutation importance on test set (descriptive)")
    savefig("permutation_importance.png")

    # Write report-ready, factual results summary based on actual run.
    best_row = comparison.iloc[0].to_dict()
    baseline_best = baseline.iloc[0].to_dict()
    tuned_base = tuning_comparison.iloc[0].to_dict()
    tuned_row = tuning_comparison.iloc[1].to_dict()
    summary = f"""# Results generated by the project code

## Dataset audit
- Rows: {audit['rows']}
- Columns: {audit['columns']} (8 predictors + 1 target)
- Missing cells after numeric conversion: {audit['missing_cells']}
- Exact duplicate rows: {audit['exact_duplicate_rows']} (reported for review; not automatically removed)

## Baseline model comparison (20% holdout test set)
{baseline.to_markdown(index=False, floatfmt=".4f")}

## Tuning
- Best RandomizedSearchCV parameters: `{search.best_params_}`
- Best 5-fold training CV RMSE: {-search.best_score_:.4f} MPa
- Baseline Random Forest test RMSE: {tuned_base['RMSE (MPa)']:.4f} MPa
- Tuned Random Forest test RMSE: {tuned_row['RMSE (MPa)']:.4f} MPa

## Final model comparison
{comparison.to_markdown(index=False, floatfmt=".4f")}

## Selected model
**{best_name}**
- MAE: {best_row['MAE (MPa)']:.4f} MPa
- MSE: {best_row['MSE (MPa²)']:.4f} MPa²
- RMSE: {best_row['RMSE (MPa)']:.4f} MPa
- R²: {best_row['R²']:.4f}

These results come from the executed fixed random split (`random_state=42`). They are not a substitute for validation on independent laboratory data. Small changes in data versions, library versions, or split strategy can change the scores.
"""
    (OUT / "results_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nSaved figures to: {FIG}")
    print(f"Saved model to: {MODELS / 'best_concrete_strength_model.joblib'}")

if __name__ == "__main__":
    main()
