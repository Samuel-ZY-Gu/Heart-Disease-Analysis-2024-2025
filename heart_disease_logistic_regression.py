from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, confusion_matrix

project_path = Path(__file__).resolve().parent

def load_year(year):
    file_path = project_path / "data" / f"Brfss_{year}_clean.CSV"
    return pd.read_csv(file_path)

df_2024 = load_year(2024)
df_2025 = load_year(2025)

common_states = set(df_2024["_STATE"].dropna().unique()) & set(df_2025["_STATE"].dropna().unique())

def prepare_data(df, common_states):
    df = df[df["_STATE"].isin(common_states)].copy()
    df = df[df["_MICHD"].isin([1, 2])].copy()
    return df

valid_2024 = prepare_data(df_2024, common_states)
valid_2025 = prepare_data(df_2025, common_states)


target = "_MICHD"

features = ["_AGEG5YR","_BMI5","_SMOKER3","_TOTINDA","DIABETE4","_EDUCAG",
            "_INCOMG1","_RFHLTH","_LTASTH1","_DRDXAR2","_RFBING6","_AIDTST4","_STATE"]

model_columns = [target] + features + ["_LLCPWT"]

model_2024 = valid_2024[model_columns].copy()
model_2025 = valid_2025[model_columns].copy()

for data in [model_2024,model_2025]:
    data[target] = data[target].map({1:1,2:0})

numeric_features = ["_BMI5"]

categorical_features = [
    col for col in features
    if col not in numeric_features
]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_train = model_2024[features]
y_train = model_2024[target]

X_test = model_2025[features]
y_test = model_2025[target]

logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight=None))
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

lr_prob = cross_val_predict(
    logistic_model, X_train, y_train,
    cv=cv, method="predict_proba"
)[:, 1]

for th in [0.05, 0.1, 0.15, 0.2, 0.3]:
    p = (lr_prob >= th).astype(int)
    print(th, round(precision_score(y_train, p, zero_division=0), 3),
          round(recall_score(y_train, p), 3))

chosen_threshold = 0.15

RUN_LOGISTIC_CV = False

if RUN_LOGISTIC_CV:

    for name, weight in [("Balanced", "balanced"), ("Unweighted", None)]:
        model = clone(logistic_model)
        model.set_params(model__class_weight=weight)

        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["average_precision", "roc_auc", "precision", "recall"]
        )

        for metric in ["average_precision", "roc_auc", "precision", "recall"]:
            print(f"{name} {metric}: {scores['test_' + metric].mean():.3f}")

forest_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=60,
        max_depth=18,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ))
])

cache_path = project_path / "rf_oof_2024.csv"

if cache_path.exists():
    rf_prob = pd.read_csv(cache_path)["probability"].to_numpy()
else:
    rf_prob = cross_val_predict(
        forest_model, X_train, y_train,
        cv=cv, method="predict_proba"
    )[:, 1]
    pd.DataFrame({"probability": rf_prob}).to_csv(cache_path, index=False)

threshold_results = []

for threshold in [0.1, 0.2, 0.3, 0.5]:
    predicted = (rf_prob >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_train, predicted, zero_division=0),
        "Recall": recall_score(y_train, predicted)
    })

pd.DataFrame(threshold_results).to_csv(
    project_path / "rf_threshold_2024.csv", index=False
)

rf_pred = (rf_prob >= 0.5).astype(int)

comparison = pd.DataFrame([
    {"Model": "Logistic Regression", "AP": average_precision_score(y_train, lr_prob),
     "ROC-AUC": roc_auc_score(y_train, lr_prob)},
    {"Model": "Random Forest", "AP": average_precision_score(y_train, rf_prob),
     "ROC-AUC": roc_auc_score(y_train, rf_prob)},
]).round(3)

print(comparison)
comparison.to_csv(project_path / "model_comparison_2024.csv", index=False)

logistic_model.fit(X_train, y_train)

test_prob = logistic_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= chosen_threshold).astype(int)

results = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Threshold": chosen_threshold,
    "AP": average_precision_score(y_test, test_prob),
    "ROC-AUC": roc_auc_score(y_test, test_prob),
    "Precision": precision_score(y_test, test_pred, zero_division=0),
    "Recall": recall_score(y_test, test_pred)
}])

print(results.round(3))
results.to_csv(project_path / "baseline_results_2025.csv", index=False)

joblib.dump(logistic_model, project_path / "heart_disease_model.joblib")

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, test_pred))