
# Banking Transaction Intelligence — Streamlit Demo

This demo showcases a machine learning case study for banking transactions moving between own-bank and other banks.

## Features
1. **% Outflow of Money** — overall and by slice (channel, MCC), with time trends.
2. **Transaction Clustering (KMeans)** — discover segments.
3. **Receiver Classification (Logistic Regression)** — predict whether a receiver is existing-to-bank or new-to-bank.
4. **Product Recommendations** — simple rule-based recommender using model outputs.

## How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

You can upload your own CSV or use the bundled synthetic data in `sample_transactions.csv`.

## Files
- `streamlit_app.py` — the Streamlit application.
- `sample_transactions.csv` — synthetic transactions for demo.
- `requirements.txt` — Python dependencies.
