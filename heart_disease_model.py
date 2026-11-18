import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from math import exp
from sklearn.inspection import permutation_importance

model = joblib.load(Path(__file__).with_name("heart_disease_model.joblib"))

logistic_regression = model.named_steps["model"]
coefficients = pd.Series(
    logistic_regression.coef_[0],
    index=model.named_steps["preprocessing"].get_feature_names_out(),
    name="coefficient",
)

numeric = model.named_steps["preprocessing"].named_transformers_["numeric"]
bmi_scale = numeric.named_steps["scaler"].scale_[0]

reference_age = coefficients["categorical___AGEG5YR_9.0"]
bmi_odds_ratio = exp(coefficients["numeric___BMI5"] * 5 / bmi_scale)
results = [f"BMI +5 odds ratio: {bmi_odds_ratio:.3f}"]

age_labels = [
    "18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
    "55–59", "60–64", "65–69", "70–74", "75–79", "80+",
]
age_odds_ratios = []

for group, age_label in enumerate(age_labels, start=1):
    age_coefficient = coefficients[f"categorical___AGEG5YR_{group}.0"]
    odds_ratio = exp(age_coefficient - reference_age)

    age_odds_ratios.append(odds_ratio)
    results.append(f"Age {age_label} vs 60–64 odds ratio: {odds_ratio:.3f}")

smoking_labels = ["Daily smoker", "Occasional smoker", "Former smoker", "Never smoker"]
reference_smoking = coefficients["categorical___SMOKER3_4.0"]
smoking_odds_ratios = []

for group, label in enumerate(smoking_labels, start=1):
    odds_ratio = exp(
        coefficients[f"categorical___SMOKER3_{group}.0"] - reference_smoking
    )
    smoking_odds_ratios.append(odds_ratio)
    results.append(f"{label} vs Never smoker odds ratio: {odds_ratio:.3f}")

pd.DataFrame({
    "variable": ["BMI"] + ["Age"] * len(age_labels) + ["Smoking"] * len(smoking_labels),
    "comparison": ["+5 units"] + age_labels + smoking_labels,
    "reference": ["5 units lower"] + ["60–64"] * len(age_labels) + ["Never smoker"] * len(smoking_labels),
    "odds_ratio": [bmi_odds_ratio] + age_odds_ratios + smoking_odds_ratios,
}).to_csv(Path(__file__).with_name("model_coefficients.csv"), index=False)

print("\n".join(results))

plt.plot(age_labels, age_odds_ratios, marker="o")
plt.axhline(1, color="gray", linestyle="--")
plt.xticks(rotation=45)
plt.xlabel("Age group")
plt.ylabel("Odds ratio vs age 60–64")
plt.tight_layout()
model_figures = Path(__file__).parent / "figures" / "model"
model_figures.mkdir(parents=True, exist_ok=True)
plt.savefig(model_figures / "age_odds_ratios.png", dpi=200)
plt.close()

project_path = Path(__file__).parent
baseline = pd.read_csv(project_path / "baseline_results_2025.csv").iloc[0]
states_2024 = pd.read_csv(
    project_path / "data" / "Brfss_2024_clean.CSV", usecols=["_STATE"]
)["_STATE"]
test_data = pd.read_csv(project_path / "data" / "Brfss_2025_clean.CSV")

test_data = test_data.loc[
    test_data["_STATE"].isin(states_2024.dropna())
    & test_data["_MICHD"].isin([1, 2])
].copy()

features = test_data[model.feature_names_in_]
positive = test_data["_MICHD"].eq(1)
detected = model.predict_proba(features)[:, 1] >= baseline["Threshold"]

age_groups = test_data["_AGEG5YR"].map(
    dict(enumerate(age_labels, start=1))
).fillna("Unknown")

age_recall = pd.DataFrame({
    "sample_size": 1,
    "positive_cases": positive,
    "true_positives": positive & detected,
}).groupby(age_groups, sort=False).sum()

age_recall = age_recall.reindex(age_labels + ["Unknown"], fill_value=0)
age_recall.loc["Overall"] = age_recall.sum()

age_recall["recall_pct"] = (
    100 * age_recall["true_positives"]
    / age_recall["positive_cases"].replace(0, float("nan"))
)

assert abs(
    age_recall.loc["Overall", "recall_pct"] / 100 - baseline["Recall"]
) < 1e-8, "Recall differs from L2"

age_recall.to_csv(project_path / "recall_by_age_2025.csv", index_label="age_group")

print(age_recall.round(2))

plt.plot(age_labels, age_recall.loc[age_labels, "recall_pct"], marker="o")
plt.axhline(age_recall.loc["Overall", "recall_pct"], color="gray", linestyle="--")
plt.xticks(rotation=45)
plt.ylabel(f"Recall (%) at threshold {baseline['Threshold']:g}")
plt.tight_layout()
plt.savefig(model_figures / "recall_by_age_2025.png", dpi=200)
plt.close()

importance = permutation_importance(
    model, features, positive,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
)

importance_table = pd.DataFrame({
    "mean_auc_drop": importance.importances_mean,
    "std_auc_drop": importance.importances_std,
}, index=features.columns).sort_values("mean_auc_drop", ascending=False)

importance_table.rename(index={
    "_AGEG5YR": "Age group",
    "_RFHLTH": "General health",
    "DIABETE4": "Diabetes",
    "_SMOKER3": "Smoking",
    "_DRDXAR2": "Arthritis",
    "_AIDTST4": "HIV testing",
    "_LTASTH1": "Asthma",
    "_STATE": "State or territory",
    "_INCOMG1": "Income",
    "_BMI5": "BMI",
    "_EDUCAG": "Education",
    "_RFBING6": "Binge drinking",
    "_TOTINDA": "Physical activity",
}, inplace=True)

importance_table.to_csv(
    project_path / "feature_importance_2025.csv", index_label="variable"
)

print(importance_table.round(4))

importance_table.iloc[::-1].plot.barh(
    y="mean_auc_drop",
    xerr="std_auc_drop",
    legend=False,
    color="teal",
    capsize=3,
)
plt.xlabel("Decrease in 2025 ROC-AUC after shuffling")
plt.ylabel("")
plt.tight_layout()
plt.savefig(model_figures / "feature_importance_2025.png", dpi=200)
plt.close()