# Heart Disease Analysis: 2024–2025

I used CDC survey data to look at how common heart disease is in the US, and whether that changed between 2024 and 2025.

The data comes from the CDC's Behavioral Risk Factor Surveillance System (BRFSS). The outcome I looked at is self-reported coronary heart disease or heart attack. I broke the results down by age, smoking, BMI, physical activity, income, education, and a few other factors.

## What I found

I compared the 49 states and territories that appear in both years' data.

| Measure | 2024 | 2025 |
|---|---|---|
| Heart disease prevalence (survey-weighted) | 6.98% | 6.70% |
| Average BMI (survey-weighted) | 28.55 | 28.52 |

Heart disease prevalence dropped by about 0.28 percentage points. This is just a description of the numbers. I didn't run significance tests, so I can't say the change is statistically meaningful, and the data can't tell us what caused it.

## What's in this repo

- `brfss_cleaning_2024.py` and `brfss_cleaning_2025.py`: clean the raw survey files
- `heart_disease_analysis_2024.py` and `heart_disease_analysis_2025.py`: analysis and charts for each year
- `heart_disease_comparison(same states).py`: compares 2024 and 2025 using only the states present in both years
- `comparison_results.csv`: the comparison results in one table
- `figures/`: all the charts

## How to run it

The raw data is too big for GitHub, so it's not included. To run the code yourself:

1. Download the 2024 and 2025 BRFSS data files from the CDC website.
2. Put them in a folder called `data/` next to the scripts.
3. Run the scripts in this order: cleaning, then the yearly analysis, then the comparison.

## Things to keep in mind

- The data is self-reported, so people may remember or answer inaccurately.
- Year-to-year comparisons only include states present in both years.
- Some survey questions and definitions changed between years (for example, how sex is defined).
- I didn't do significance testing or adjust for other factors, so treat these as descriptive results.

## Tools

Python, pandas, NumPy, Matplotlib