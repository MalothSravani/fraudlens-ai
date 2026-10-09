# FraudLens AI — Transaction Risk Intelligence

A polished, interactive **educational fraud-screening prototype** built with Streamlit, pandas, scikit-learn, and Plotly. It demonstrates a transaction-risk workflow: data inspection, class-imbalanced classification, risk prioritization, threshold trade-offs, and model evaluation.

> **Important:** The app uses clearly labelled simulated transactions by default. Synthetic results are not real-world fraud rates or evidence of production performance. The app is not a production fraud-control system and must not be connected to live payment authorization.

## Features

- Executive overview with risk distribution and evaluation metrics
- Interactive transaction scoring form
- Precision, recall, PR-AUC, ROC-AUC, confusion matrix, and threshold analysis
- Data explorer with downloadable scored rows
- Optional upload of a compact labelled CSV
- SQL schema and example analysis queries
- Streamlit deployment-ready structure

## Run locally

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Use a real labelled dataset

The app accepts a labelled CSV with a fraud target named one of `isFraud`, `is_fraud`, `fraud`, `target`, or `label`, plus an amount column named `TransactionAmt` or `amount`. It maps a small set of common IEEE-CIS columns where available. For the large IEEE-CIS files, first make a manageable sample locally; do not upload the entire competition dataset to a free demo app.

Dataset source: [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Kaggle requires account access and agreement to competition rules. Check those rules before redistributing any data. A practical first experiment is a labelled sample of 10,000–30,000 rows.

For example, locally create a compact sample from `train_transaction.csv`:

```python
import pandas as pd
cols = ["TransactionID", "TransactionAmt", "TransactionDT", "ProductCD", "card4", "DeviceType", "isFraud"]
df = pd.read_csv("train_transaction.csv", usecols=lambda c: c in cols, nrows=30000)
df.to_csv("ieee_cis_sample.csv", index=False)
```

If using the Kaggle data, verify the actual columns and sample class balance before uploading. The app's simple random holdout is for demonstration only; a research-grade result should use chronological validation where timestamps permit it and should explicitly test for leakage.

## Deploy a live website

1. Create a GitHub repository named `fraudlens-ai`.
2. Upload this folder's contents to the repository root (including `app.py`, `requirements.txt`, and `.streamlit/config.toml`).
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
4. Select **Create app**, choose the repository and `app.py`, then click **Deploy**.
5. Once deployment completes, copy the actual URL ending in `streamlit.app`.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

I cannot publish to your GitHub or Streamlit account on your behalf without access and authorization, so a personal live URL is not created by generating this code bundle alone.

## Project structure

```text
fraudlens_ai/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── sql/schema.sql
├── sql/analysis_queries.sql
└── tests/test_demo_data.py
```

## Model notes

- Model: class-weighted Random Forest with imputation and categorical encoding
- Default split: stratified random holdout for demo convenience
- Metrics: precision, recall, PR-AUC, ROC-AUC, confusion matrix
- Default review bands: Low < 35%, Review 35–70%, High ≥ 70%
- Thresholds are illustrative and not calibrated to actual investigation costs

## Before using this on a resume

- Deploy and test the app in a clean environment.
- Run the project with a real, permitted dataset and document exactly which rows/features were used.
- Report measured evaluation metrics; do not invent performance claims.
- Add time-aware validation, threshold calibration, and model limitations if presenting results as research.
- Do not describe the synthetic demo as a model trained on real fraud data.

## License

Choose a license for your own code before publishing. Dataset rights and terms remain separate; follow the dataset provider's rules.
