# US Company Bankruptcy Prediction

Predicting which US public companies (NYSE and NASDAQ, 1999-2018) are at risk of filing for bankruptcy in the following year, based on their financial statements.

**Live demo:** _add your Streamlit link here_

## Key Findings

1. **Label issue.** `status_label` is `failed` for every year of a company that eventually failed. The target was rebuilt as the last observed year of each failed company (609 cases, 0.77% of rows). The yearly counts match those reported in the source paper.
2. **Data leakage.** Using the original label, a random row split gives a ROC-AUC of 0.851 and a PR-AUC of 0.441. Splitting by company gives only 0.690 and 0.144.
3. **Model performance.** On unseen years (2015-2018), inspecting the riskiest 10% of companies in each year catches about 75% of bankruptcies.

## Method

- **Features:** financial ratios (including Altman Z-score components) and year-over-year changes
- **Time-based split:** train 1999-2011, validation 2012-2014, test 2015-2018
- **Models:** Logistic Regression, Random Forest, LightGBM, and the Altman Z-score as a baseline
- **Selection:** best validation PR-AUC, then retrained on 1999-2014

## Test Set Results

| Model                     | PR-AUC    | ROC-AUC   | Recall@10% |
| ------------------------- | --------- | --------- | ---------- |
| **Random Forest**         | **0.330** | **0.932** | **0.748**  |
| LightGBM                  | 0.286     | 0.929     | 0.748      |
| Logistic Regression       | 0.156     | 0.899     | 0.706      |
| Altman Z-score (baseline) | 0.026     | 0.794     | 0.227      |

A random model would score a PR-AUC of 0.0097.

_Recall@10%: share of bankruptcies caught when inspecting the riskiest 10% of companies in each year._

## Most Important Features

Based on permutation importance on the validation set (model trained on 1999-2011):

1. Market value to total liabilities
2. Total assets (log)
3. Market value (log)
4. Quick ratio

## Dashboard

- **Risk Ranking:** the riskiest companies for a selected year
- **Company:** risk score history of a single company
- **Model Performance:** comparison of all models

## Project Structure

```
├── app.py                       # Streamlit dashboard
├── Bankruptcy Prediction.ipynb  # Analysis and modeling     
├── scores.csv                   # Test set risk scores
├── metrics.csv                  # Model comparison
├── requirements.txt             # Dashboard dependencies
└── README.md
```

## Getting Started

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The notebook also requires `numpy`, `matplotlib`, `scikit-learn`, and `lightgbm`. The raw dataset is not included; set `DATA_PATH` in the notebook to your local copy.

## Limitations

- Precision is low (about 7% in the top 10%), so the model is suited for initial screening only.
- Random Forest and LightGBM were nearly tied on validation PR-AUC (0.219 vs 0.218).
- Risk scores are not calibrated probabilities.
- Some raw values look unusual (e.g., negative sales) and were left unchanged.
- Feature importance does not imply causation.

## Data Source

Lombardo, G., et al. (2022). _Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks_. Future Internet, 14(8), 244.
[Dataset repository](https://github.com/sowide/bankruptcy_dataset)
