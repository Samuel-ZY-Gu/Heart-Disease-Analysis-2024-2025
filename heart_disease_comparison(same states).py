import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import us

project_path = Path(__file__).resolve().parent

data_2024 = pd.read_csv(project_path / "data" / "Brfss_2024_clean.CSV")
data_2025 = pd.read_csv(project_path / "data" / "Brfss_2025_clean.CSV")
print("2024 rows:",len(data_2024))
print("2025 rows:",len(data_2025))

common_states = set(data_2024["_STATE"].dropna().unique()) & set(data_2025["_STATE"].dropna().unique())

common_2024 = data_2024[data_2024["_STATE"].isin(common_states)]
common_2025 = data_2025[data_2025["_STATE"].isin(common_states)]

print(f"Common areas: {len(common_states)}, 2024 rows: {len(common_2024):,}, 2025 rows: {len(common_2025):,}")

valid_2024 = common_2024[common_2024["_MICHD"].isin([1, 2])]
valid_2025 = common_2025[common_2025["_MICHD"].isin([1, 2])]

def weighted_prevalence(data):
    return (
        data.loc[data["_MICHD"] == 1, "_LLCPWT"].sum()
        / data["_LLCPWT"].sum() * 100)

prevalence_2024 = weighted_prevalence(valid_2024)
prevalence_2025 = weighted_prevalence(valid_2025)
print(f"2024 Weighted Prevalence: {prevalence_2024:.2f}%")
print(f"2025 Weighted Prevalence: {prevalence_2025:.2f}%")

change_pp = prevalence_2025 - prevalence_2024
print(f"Change (percentage points): {change_pp:+.2f}")

def grouped_prevalence(data, column):
    total = data.groupby(column)["_LLCPWT"].sum()
    cases = data.loc[data["_MICHD"] == 1].groupby(column)["_LLCPWT"].sum()
    return cases / total * 100

group_variables = {
    "Age": "_AGEG5YR",
    "Smoking": "_SMOKER3",
    "BMI": "_BMI5CAT",
    "Physical Activity": "_TOTINDA",
    "Diabetes": "DIABETE4",
    "Sex": "_SEX",
    "Education": "_EDUCAG",
    "Income": "_INCOMG1",
    "General Health": "_RFHLTH",
    "Asthma": "_LTASTH1",
    "Arthritis": "_DRDXAR2",
    "Binge Drinking": "_RFBING6",
    "HIV Testing": "_AIDTST4",
    "State": "_STATE"
}

comparisons = {}

for name, column in group_variables.items():
    result = pd.DataFrame({
        "2024 (%)": grouped_prevalence(valid_2024, column),
        "2025 (%)": grouped_prevalence(valid_2025, column)
    })

    result["Change (pp)"] = result["2025 (%)"] - result["2024 (%)"]

    comparisons[name] = result

    print(f"\n{name} Comparison:\n", result.round(2))

bmi_means = {}

for year, data in {2024: common_2024, 2025: common_2025}.items():
    valid_bmi = data.dropna(subset=["_BMI5", "_LLCPWT"])
    bmi_means[year] = (valid_bmi["_BMI5"] * valid_bmi["_LLCPWT"]).sum() / valid_bmi["_LLCPWT"].sum()

print("Weighted Mean BMI:", {year: round(value, 2) for year, value in bmi_means.items()})
print(f"BMI Change: {bmi_means[2025] - bmi_means[2024]:+.2f}")

figures_path = project_path / "figures" / "comparison"
figures_path.mkdir(parents=True, exist_ok=True)

label_maps = {
    "Age": ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
            "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
    "Smoking": ["Daily", "Some Days", "Former", "Never"],
    "BMI": ["Underweight", "Normal", "Overweight", "Obese"],
    "Physical Activity": ["Any Leisure Activity", "No Leisure Activity"],
    "Diabetes": ["Diabetes", "Pregnancy Only", "No Diabetes", "Prediabetes"],
    "Sex": ["Male", "Female"],
    "Education": ["Below HS", "HS Graduate", "Some College/Tech", "College/Tech Graduate"],
    "Income": ["<15k", "15-25k", "25-35k", "35-50k", "50-100k", "100-200k", "200k+"],
    "General Health": ["Good or Better", "Fair or Poor"],
    "Asthma": ["No", "Yes"],
    "Arthritis": ["Yes", "No"],
    "Binge Drinking": ["No", "Yes"],
    "HIV Testing": ["Yes", "No"]
}

for name, result in comparisons.items():
    if name == "State":
        continue

    plot_data = result[["2024 (%)", "2025 (%)"]].copy()
    plot_data.index = label_maps[name]

    plot_data.plot(kind="bar", figsize=(8, 5), rot=0)

    title = f"{name} Comparison: 2024 vs 2025"
    if name == "Sex":
        title += " (Definition Changed)"
    plt.title(title)
    plt.xlabel(name)
    plt.ylabel("Weighted Prevalence (%)")
    plt.legend(title=None)
    plt.tight_layout()

    file_name = name.lower().replace(" ", "_") + "_comparison.png"
    plt.savefig(figures_path / file_name, dpi=300, bbox_inches="tight")
    plt.show()

state_plot = comparisons["State"]["Change (pp)"].copy()
state_plot.index = [us.states.lookup(str(int(code)).zfill(2)).abbr for code in state_plot.index]
state_plot.sort_values().plot(kind="barh", figsize=(9, 12))

plt.axvline(0, color="black", linewidth=0.8)
plt.title("Change in Reported CHD or MI Prevalence by Area (2024–2025)")
plt.xlabel("Change (percentage points)")
plt.ylabel("State / Territory")
plt.tight_layout()
plt.savefig(figures_path / "state_change_comparison.png", dpi=300)
plt.show()

pd.Series(bmi_means).plot(kind="bar", figsize=(5, 4), rot=0)
plt.title("Weighted Mean BMI: 2024 vs 2025")
plt.xlabel("Year")
plt.ylabel("Weighted Mean BMI")
plt.ylim(0, 35)
plt.tight_layout()
plt.savefig(figures_path / "bmi_mean_comparison.png", dpi=300)
plt.show()

pd.concat(comparisons, names=["Variable", "Category Code"]).to_csv(
    project_path / "comparison_results.csv")