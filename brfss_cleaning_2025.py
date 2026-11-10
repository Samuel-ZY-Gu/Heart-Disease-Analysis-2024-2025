from pathlib import Path
import pandas as pd

project_path = Path(__file__).resolve().parent
file_path = project_path / "data" / "LLCP2025.XPT"
out_path = project_path / "data" / "Brfss_2025_clean.CSV"

target_columns = [
    "_MICHD", "_AGEG5YR", "_BMI5", "_SMOKER3", "_TOTINDA","_SEX", "_EDUCAG", "_INCOMG1",
    "_RFHLTH","_RFHYPE6","DIABETE4","_STATE", "_LLCPWT","_STSTR", "_PSU","_LTASTH1","_DRDXAR2",
    "_BMI5CAT","_RFBING6","_FRUTSU1","_VEGESU1","_AIDTST4"
]

reader = pd.read_sas(file_path, format= "xport",chunksize=(400000))
df = pd.concat((chunk[target_columns] for chunk in reader), ignore_index=True)

df_selected = df[target_columns].copy()

numeric_columns = ["_BMI5", "_FRUTSU1", "_VEGESU1"]
survey_columns = ["_STATE", "_LLCPWT", "_STSTR", "_PSU"]

categorical_columns = [
    col for col in target_columns
    if col not in numeric_columns + survey_columns
]
print("Numerical:", numeric_columns,"\nSurvey:", survey_columns,"\nCategorical:", categorical_columns)

df_numeric_clean = df_selected.copy()
for col in numeric_columns:
    df_numeric_clean[col] = df_numeric_clean[col]/100

bmi_invalid = (df_numeric_clean["_BMI5"] < 12) | (df_numeric_clean["_BMI5"] >= 100)
df_numeric_clean.loc[bmi_invalid, "_BMI5"] = float("nan")

limits = {"_FRUTSU1": 16,"_VEGESU1": 23}
for col, limit in limits.items():
    invalid = df_numeric_clean[col] > limit
    df_numeric_clean.loc[invalid, col] = float("nan")
    print(col, "Cleaned:", invalid.sum())

for col in limits:
    near_zero = (df_numeric_clean[col] > 0) & (df_numeric_clean[col] < 1e-10)
    df_numeric_clean.loc[near_zero, col] = 0

print("\nFinal numerical summary:")
print(df_numeric_clean[numeric_columns].describe())

print("\nsurvey unique values:")
print(df_numeric_clean[survey_columns].nunique())

print("\nsurvey weight values:")
print(df_numeric_clean["_LLCPWT"].describe())

df_clean = df_numeric_clean.copy()

special_codes = {
    "_AGEG5YR": [14],"_SMOKER3": [9],"_TOTINDA": [9],"_EDUCAG": [9],
    "_INCOMG1": [9],"_RFHLTH": [9],"_RFHYPE6": [9],"DIABETE4": [7, 9],
    "_LTASTH1": [9], "_RFBING6": [9],"_AIDTST4": [9]
}

for col, codes in special_codes.items():
    df_clean[col] = df_clean[col].replace(codes, float("nan"))

valid_codes = {"_MICHD": [1, 2],"_DRDXAR2": [1, 2],"_BMI5CAT": [1, 2, 3, 4],"_SEX": [1, 2]}

assert df_clean.shape == df_selected.shape

print("Final shape:", df_clean.shape)

missing = df_clean.isna().sum()
print("Missing values:\n", missing[missing > 0])

assert df_clean["_BMI5"].dropna().between(12, 100, inclusive="left").all()

for col, limit in limits.items():
    assert df_clean[col].dropna().between(0, limit).all()

for col, codes in special_codes.items():
    assert not df_clean[col].isin(codes).any()

for col, codes in valid_codes.items():
    assert df_clean[col].dropna().isin(codes).all()

assert (df_clean["_LLCPWT"] > 0).all()

for col in ["_STATE", "_STSTR", "_PSU"]:
    assert df_clean[col].notna().all()

expected_cat = pd.cut(df_clean["_BMI5"], bins=[0, 18.5, 25, 30, 100], labels=[1, 2, 3, 4], right=False)
assert (expected_cat.astype(float).fillna(0) == df_clean["_BMI5CAT"].fillna(0)).all()

print("Basic Data Validation Passed!")
df_clean.to_csv(out_path, index=False)