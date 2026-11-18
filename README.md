# Heart Disease Analysis: 2024–2025

A personal data project using CDC health survey data. I started with two questions.

1. Did self-reported heart disease become more or less common between 2024 and 2025?
2. Can basic health and lifestyle answers help pick out people who are more likely to report heart disease?

I used Python to clean the data, compare the two years, make charts, and build a prediction model. After comparing models, I looked at how the selected model uses age, BMI and smoking, and how many cases it catches in each age group.

The data comes from the CDC's Behavioral Risk Factor Surveillance System (BRFSS), a yearly health survey. In this project, "heart disease" means the respondent reported coronary heart disease or a heart attack.

## 1. Did heart disease change between 2024 and 2025?

I set answers like "don't know" and "refused" to missing, removed impossible BMI values, and kept the variables needed for the analysis. For the year-to-year comparison, I used the 49 states and territories that appear in both datasets.

| Measure | 2024 | 2025 |
|---|---|---|
| Heart disease prevalence | 6.98% | 6.70% |
| Average BMI | 28.55 | 28.52 |

Both numbers use survey weights. Reported heart disease fell by about 0.28 percentage points, while average BMI barely changed. I also compared age groups, smoking, BMI categories, physical activity, income, education, and other survey variables. The charts are in `figures/`.

These are descriptive comparisons. I haven't tested whether the changes are statistically significant or what caused them.

## 2. Can a model pick out who reports heart disease?

I trained on 437,288 valid responses from 2024 and evaluated on 346,500 from 2025, using the same 49 areas. The model uses 13 variables, covering age, BMI, smoking, physical activity, diabetes, education, income, general health, asthma, arthritis, binge drinking, HIV testing, and state.

I compared logistic regression and random forest using 3-fold stratified cross-validation on 2024. Missing values were handled inside each training fold. I then searched random forest parameters, selecting configurations by average precision (AP).

I evaluated 28 forest configurations, varying tree depth, minimum leaf size, feature subsampling, class weighting, bootstrap sample fraction and tree count.

| Model | 2024 cross-validation AP | 2024 cross-validation ROC-AUC |
|---|---|---|
| Logistic regression | 0.2968 | 0.8089 |
| Original random forest | 0.2924 | 0.8056 |
| Best random forest in this search | 0.2933 | 0.8061 |

Both metrics measure how well the models rank respondents by their likelihood of reporting heart disease. Random ranking gives a ROC-AUC around 0.5, with AP around the proportion of positive cases, roughly 0.09 here.

The best forest kept the original maximum depth (18) and minimum leaf size (10), with 400 trees instead of 60. Increasing the tree count slightly raised cross-validation AP from 0.2924 to 0.2933. Logistic regression still scored slightly higher, so I kept it as the main model. Its coefficients are also easier to explain. Configuration rankings and evaluation results are in `l3/rf_search/`.

### Choosing a cutoff

I tried several cutoffs using the 2024 out-of-fold predictions and selected 0.15 for logistic regression. At that cutoff, cross-validation caught about 58% of reported cases, and about 1 in 4 flagged respondents had reported heart disease. Lowering the cutoff catches more cases and produces more false alarms.

### Results on 2025

| Model | Cutoff | AP | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic regression | 0.15 | 0.2823 | 0.8076 | 24.8% | 59.2% |
| Best random forest in this search | 0.15 | 0.2808 | 0.8054 | 24.1% | 60.9% |

Of the 346,500 respondents in the evaluation sample, 31,089 reported heart disease. Logistic regression caught 18,402 of them and flagged 55,721 people who did not report it.

At the same 0.15 cutoff, random forest caught 527 more cases and produced 4,008 more false alarms. I also chose a forest cutoff of about 0.1545 using only 2024 predictions to match logistic regression's 2024 recall. On 2025, the forest reached 59.25% recall with 24.49% precision. At that similar recall, its precision was still slightly below logistic regression's 24.83%, which came with 59.19% recall.

What surprised me was that a ROC-AUC around 0.81 could still come with so many false alarms. About three out of four people flagged by logistic regression had not reported heart disease.

## 3. What does the logistic model use, and who does it miss?

### Model coefficients

I converted the fitted coefficients into odds ratios. These comparisons hold the other model inputs fixed.

| Comparison | Odds ratio |
|---|---|
| BMI 5 units higher | 1.03 |
| Age 80+ compared with age 60-64 | 2.88 |
| Daily smoker compared with never smoker | 1.70 |
| Occasional smoker compared with never smoker | 1.68 |
| Former smoker compared with never smoker | 1.52 |

An odds ratio of 1.70 means the model estimates 1.70 times the odds for that comparison. Odds are calculated as probability divided by one minus probability. The coefficient table uses age 60-64 and never smokers as reference groups. These are fitted associations with self-reported heart disease.

The full BMI, age and smoking comparisons are in `model_coefficients.csv`.

### Feature importance

I shuffled each of the 13 original variables separately in the 2025 evaluation data and measured the drop in ROC-AUC, repeating each shuffle five times. Age caused the largest drop, followed by general health and diabetes. Smoking ranked fourth.

This measures how much the fitted model relies on each variable for ranking respondents. Correlated variables can share information, which affects their individual rankings.

The means and standard deviations are in `feature_importance_2025.csv`. The ranking chart is in `figures/model/`.

### Recall by age

The overall recall of 59.2% hides large differences between age groups. All groups below use the same 0.15 cutoff and unweighted sample counts.

| Age group | Respondents | Reported cases | Cases caught | Recall |
|---|---|---|---|---|
| 18-24 | 22,585 | 164 | 0 | 0.0% |
| 25-29 | 17,514 | 139 | 0 | 0.0% |
| 30-34 | 18,839 | 215 | 0 | 0.0% |
| 35-39 | 20,972 | 343 | 5 | 1.5% |
| 40-44 | 22,455 | 513 | 25 | 4.9% |
| 45-49 | 21,513 | 740 | 121 | 16.4% |
| 50-54 | 22,597 | 1,237 | 405 | 32.7% |
| 55-59 | 25,328 | 1,923 | 906 | 47.1% |
| 60-64 | 31,751 | 3,094 | 1,612 | 52.1% |
| 65-69 | 36,201 | 4,243 | 2,248 | 53.0% |
| 70-74 | 34,773 | 5,031 | 2,990 | 59.4% |
| 75-79 | 30,164 | 5,394 | 3,810 | 70.6% |
| 80+ | 35,465 | 7,601 | 6,137 | 80.7% |
| Age missing | 6,343 | 452 | 143 | 31.6% |
| Overall | 346,500 | 31,089 | 18,402 | 59.2% |

Recall is cases caught divided by reported cases. The model missed all 518 reported cases among respondents aged 18-34 at this cutoff. For respondents aged 80 and older, it caught about 81%. The full results are in `recall_by_age_2025.csv`.

## 4. Files

| File | What it does |
|---|---|
| `brfss_cleaning_2024.py`, `brfss_cleaning_2025.py` | Standalone cleaning scripts for each year |
| `heart_disease_analysis_2024.py`, `heart_disease_analysis_2025.py` | Standalone analysis and charts for each year |
| `heart_disease_comparison.py` | Compares the two years |
| `heart_disease_logistic_regression.py` | Trains and evaluates the baseline models |
| `heart_disease_model.py` | Explains logistic regression and evaluates recall by age |
| `comparison_results.csv` | Year-to-year comparison results |
| `model_comparison_2024.csv` | Original model comparison on 2024 |
| `baseline_results_2025.csv` | Logistic regression evaluation on 2025 |
| `model_coefficients.csv` | BMI, age and smoking odds ratios |
| `feature_importance_2025.csv` | Permutation importance rankings |
| `recall_by_age_2025.csv` | Age-group sample sizes, case counts and recall |
| `figures/` | Charts, including model plots in its model subfolder |
| `l3/tune_random_forest.py` | Runs the separate random forest parameter search |
| `l3/rf_search/` | Configuration rankings, selected parameters, threshold comparisons and final evaluation |
| `requirements.txt` | Python packages needed |

## 5. How to run it

The original BRFSS files are too large for GitHub, so they are not included.

1. Install the packages.
   ```
   pip install -r requirements.txt
   ```
2. Download the 2024 and 2025 BRFSS data (SAS transport files) from the CDC website and save them in a `data/` folder as `LLCP2024.XPT` and `LLCP2025.XPT`.
3. From the project root, run the scripts in this order.

   ```sh
   python brfss_cleaning_2024.py
   python brfss_cleaning_2025.py
   python heart_disease_analysis_2024.py
   python heart_disease_analysis_2025.py
   python heart_disease_comparison.py
   python heart_disease_logistic_regression.py
   python heart_disease_model.py
   ```

The baseline training script reuses `rf_oof_2024.csv` when it exists. If the training data, row order, preprocessing or forest parameters change, the cached predictions need to be regenerated.

The additional forest search runs separately.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python l3/tune_random_forest.py
```

It saves progress under `l3/rf_search/` and reuses completed candidates when input hashes, search code and recorded environment versions match. The repository includes configuration rankings, selected parameters, threshold comparisons and the final evaluation. Per-respondent predictions, fitted models and candidate caches stay local.

## 6. Limitations

- The target comes from self-reported coronary heart disease or heart attack. The model has no clinical validation for diagnosis or individual treatment decisions, and the survey provides no follow-up outcome for predicting future onset.
- I used 2024 for training and cutoff selection, but had already looked at some 2025 results while comparing models. That limits how independently the 2025 evaluation can confirm performance.
- The survey is a snapshot. General health and other answers may reflect an existing condition, so their coefficients cannot establish what caused it.
- Survey weights are used in the descriptive analysis. Model training and evaluation use unweighted samples.
- Some questions and definitions changed between years, including how sex is defined. The year-to-year comparison covers only areas present in both datasets.
- I haven't formally evaluated probability calibration or tested the model on data outside these BRFSS samples.
- The forest search identifies the best of 28 evaluated configurations. The same cross-validation results were used to select parameters, so those search scores include selection bias.
- Age-group recall varies sharply at the chosen cutoff. Smaller case counts in younger groups also make those estimates more sensitive to individual predictions.

## What I learned

Cleaning took more work than I expected, especially checking missing-value codes and making the two years comparable. Survey weights also changed the picture. About 9% of respondents in the evaluation sample reported heart disease, compared with a weighted estimate of 6.7%.

Tuning random forest gave me a firmer reason to keep logistic regression. Breaking recall down by age was more troubling. The model caught a majority of cases overall while missing every reported case among respondents under 35.

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, joblib
