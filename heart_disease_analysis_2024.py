import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def calculate_prevalence(data, column):
    table = data.groupby(column)["_MICHD"].value_counts().unstack(fill_value=0)
    table["Prevalence (%)"] = table[1.0] / table.sum(axis=1) * 100

    total = data.groupby(column)["_LLCPWT"].sum()
    cases = data.loc[data["_MICHD"] == 1].groupby(column)["_LLCPWT"].sum()
    table["Weighted Prevalence (%)"] = cases / total * 100

    return table

project_path = Path(__file__).resolve().parent
file_path = project_path / "data" / "Brfss_2024_clean.CSV"

df = pd.read_csv(file_path)
print("Data shape:", df.shape)

heart = df["_MICHD"].dropna()
prevalence = (heart == 1).mean() * 100
print("Heart disease prevalence:", prevalence)
print("Valid responses:", len(heart))
print("Heart disease cases:", (heart == 1).sum())

valid = df[df["_MICHD"].isin([1, 2])]
weighted = valid.loc[valid["_MICHD"] == 1, "_LLCPWT"].sum() / valid["_LLCPWT"].sum() * 100
print(f"Weighted prevalence: {weighted:.2f}%")

figures_path = project_path / "figures" / "2024"
figures_path.mkdir(parents=True, exist_ok=True)

variables = {
    "Age Group": ("_AGEG5YR", [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
    ], "age"),
    "Smoking Status": ("_SMOKER3", [
        "Daily smoker", "Occasional smoker", "Former smoker", "Never smoker"
    ], "smoking"),
    "BMI Category": ("_BMI5CAT", [
        "Underweight", "Normal weight", "Overweight", "Obese"
    ], "bmi"),
    "Physical Activity": ("_TOTINDA", [
        "Activity reported", "No activity reported"
    ], "activity"),
    "Diabetes Status": ("DIABETE4", [
        "Diabetes", "Gestational only", "No diabetes", "Prediabetes"
    ], "diabetes")
}

for name, (column, labels, filename) in variables.items():
    table = calculate_prevalence(valid, column)
    table.index = labels

    print(f"\n{name} Analysis:\n", table.round(2))

    plot_data = table[["Prevalence (%)", "Weighted Prevalence (%)"]]

    if name == "Age Group":
        plot_data.plot(marker="o", figsize=(10, 5))
    else:
        plot_data.plot(kind="bar", figsize=(9, 5),
                       rot=15 if name in ["Smoking Status", "Diabetes Status"] else 0)

    plt.title(f"Reported CHD or MI Prevalence by {name} (2024)")
    plt.xlabel(name if name != "Physical Activity" else "Physical Activity (Past 30 Days)")
    plt.ylabel("Prevalence (%)")
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(figures_path / f"{filename}_prevalence_2024.png",
                dpi=300, bbox_inches="tight")
    plt.show()

