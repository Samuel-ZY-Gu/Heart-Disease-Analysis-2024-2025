# Heart Disease Analysis: 2024–2025

A personal data project using CDC health survey data. I wanted to answer two questions:

1. Did self-reported heart disease become more or less common between 2024 and 2025?
2. Can basic health and lifestyle answers help pick out people who are more likely to report heart disease?

I used Python to clean the data, compare the two years, make charts, and build a simple prediction model.

The data comes from the CDC's Behavioral Risk Factor Surveillance System (BRFSS), a yearly health survey. In this project, "heart disease" means the respondent said they had coronary heart disease or a heart attack.

## 1. Did heart disease change between 2024 and 2025?

First I cleaned both years. I set answers like "don't know" and "refused" to missing, removed impossible BMI values, and kept only the variables I needed. For the year-to-year comparison I only used the 49 states and territories that appear in both datasets.

| Measure | 2024 | 2025 |
|---|---|---|
| Heart disease prevalence | 6.98% | 6.70% |
| Average BMI | 28.55 | 28.52 |

Both numbers use survey weights. Reported heart disease fell by about 0.28 percentage points, and average BMI barely moved. I also compared age groups, smoking, BMI categories, physical activity, income, education, and a few other factors (the charts are in `figures/`).

These are descriptive comparisons. I didn't test whether the changes are statistically significant, and they can't tell us what caused them.

## 2. Can a model pick out who reports heart disease?

I trained the model on the 2024 data and tested it on 2025. It uses 13 survey answers: age, BMI, smoking, physical activity, diabetes, education, income, general health, asthma, arthritis, binge drinking, HIV testing, and state.

**Choosing a model.** I compared logistic regression and random forest using 3-fold cross-validation on the 2024 data.

| Model | Average precision | ROC-AUC |
|---|---|---|
| Logistic regression | 0.297 | 0.809 |
| Random forest | 0.292 | 0.806 |

Both columns measure how well a model ranks people from least to most likely to report heart disease. Random guessing would give a ROC-AUC of 0.5 and an average precision of about 0.09. The two models came out almost the same, so I kept logistic regression because it is simpler and easier to explain.

**Choosing a cutoff.** The model gives each person a score, so I needed a cutoff for "flagged as likely." I tried several on the 2024 data and picked 0.15. In cross-validation that caught about 58% of cases, and about 1 in 4 flagged people really had reported heart disease. A lower cutoff catches more cases but flags more people by mistake, and a higher one does the opposite.

**Testing on 2025.**

| Average precision | ROC-AUC | Precision | Recall |
|---|---|---|---|
| 0.282 | 0.808 | 24.8% | 59.2% |

The 2025 test data has 346,500 valid responses, and 31,089 of them (about 9%) reported heart disease. The model caught 18,402 of those people, but it also flagged 55,721 people who did not report it.

What surprised me most is that a ROC-AUC of 0.81 sounds good, yet about three out of four flagged people were wrong. When only about 9% of people have the condition, a model can rank people well and still make a lot of false alarms. That taught me to look at the cutoff and the counts, not just one summary number.

## 3. Files

| File | What it does |
|---|---|
| `brfss_cleaning_2024.py`, `brfss_cleaning_2025.py` | Clean the raw survey data |
| `heart_disease_analysis_2024.py`, `heart_disease_analysis_2025.py` | Analysis and charts for each year |
| `heart_disease_comparison.py` | Compares the two years |
| `heart_disease_logistic_regression.py` | Trains and evaluates the models |
| `comparison_results.csv` | Year-to-year comparison results |
| `model_comparison_2024.csv` | Model comparison on 2024 |
| `baseline_results_2025.csv` | Final evaluation on 2025 |
| `figures/` | Charts |
| `requirements.txt` | Python packages needed |

## 4. How to run it

The original BRFSS files are too large for GitHub, so they are not included.

1. Install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Download the 2024 and 2025 BRFSS data (SAS transport files) from the CDC website and save them in a `data/` folder as `LLCP2024.XPT` and `LLCP2025.XPT`.
3. Run the cleaning scripts first, then the yearly analysis scripts, then the comparison, then the model script.

## 5. Limitations

- **This is not a diagnosis.** The model finds patterns in self-reported survey answers. It does not predict who will develop heart disease in the future.
- **The 2025 data was not completely untouched.** I used 2024 for training and for choosing the cutoff, but I also looked at some 2025 results while comparing models. Read the final test with that in mind.
- **The survey is a one-time snapshot.** For example, people with heart disease may report worse general health, but that doesn't mean poor health caused their condition.
- **Survey weights were not used in the model.** I used them in the descriptive analysis, but the model evaluation uses the unweighted sample.
- **The two years are not perfectly identical.** Some questions and definitions changed (for example, how sex is defined), and the comparison only covers areas in both datasets.
- **There is more to check.** I haven't formally evaluated probability calibration, so the model's scores should not be treated as reliable estimates of individual medical risk. I also haven't tested the model on any outside data.

## What I learned

- Real survey data takes a lot of cleaning and checking. Most of the code in this project is cleaning, not modeling.
- Weights change the picture. About 9% of people in the test sample reported heart disease, but the weighted estimate is 6.7%.
- Running a model is the easy part. Choosing a fair way to compare models, picking a cutoff, and being honest about the limits took more thought.

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn