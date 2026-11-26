from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bankruptcy Risk Dashboard", layout="wide")
BASE = Path(__file__).parent  # folder that contains this file


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "scores.csv"), pd.read_csv(BASE / "metrics.csv")


scores, metrics = load_data()

st.title("US Company Bankruptcy Risk Dashboard")
st.caption("Test period 2015-2018. Scores rank relative risk; they are not calibrated probabilities.")

tab_rank, tab_company, tab_model = st.tabs(["Risk Ranking", "Company", "Model Performance"])

with tab_rank:
    year = st.selectbox("Year", sorted(scores.year.unique(), reverse=True))
    pct = st.slider("Share of companies to inspect (%)", 1, 30, 10)

    data = scores[scores.year == year].sort_values("risk_rank")
    n = max(int(len(data) * pct / 100), 1)
    flagged = data.head(n)
    caught = int(flagged.target.sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Companies flagged", n)
    c2.metric("Bankruptcies caught", f"{caught} of {int(data.target.sum())}")
    c3.metric("Precision", f"{caught / n:.1%}")

    table = flagged[["risk_rank", "company_name", "risk_score", "target"]]
    st.dataframe(table.rename(columns={"target": "actually_bankrupt"}), hide_index=True)

with tab_company:
    company = st.selectbox("Select a company", sorted(scores.company_name.unique()))
    history = scores[scores.company_name == company].sort_values("year")

    st.subheader("Risk score by year")
    st.line_chart(history.assign(year=history.year.astype(str)), x="year", y="risk_score")
    st.dataframe(history.rename(columns={"target": "actually_bankrupt"}), hide_index=True)

with tab_model:
    st.subheader("Model comparison on the test set")
    st.dataframe(metrics.round(3), hide_index=True)
    st.bar_chart(metrics, x="model", y="PR-AUC")
