import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLB Salary Value Explorer", layout="wide")

@st.cache_data
def load_outputs():
    bargains = pd.read_csv("top_bargains_2023.csv")
    overpays = pd.read_csv("top_overpays_2023.csv")
    return bargains, overpays

st.title("MLB Salary Value Explorer — 2023")
st.caption("Predicts year‑t salary from t−1 performance + context; shows top bargains/overpays.")

bargains, overpays = load_outputs()

teams = sorted(set(bargains['team'].dropna().unique()) | set(overpays['team'].dropna().unique()))
team_sel = st.sidebar.multiselect("Filter by team", teams, default=[])

def filt(df):
    return df[df['team'].isin(team_sel)] if team_sel else df

col1, col2 = st.columns(2)
fmt = lambda s: s.round(0).map('${:,.0f}'.format)

with col1:
    st.subheader("Top Bargains — 2023")
    tbl = filt(bargains).assign(
        pred_salary=lambda d: fmt(d['pred_salary']),
        salary_aav_usd=lambda d: fmt(d['salary_aav_usd']),
        residual=lambda d: fmt(d['residual'])
    )
    st.dataframe(tbl, use_container_width=True)
    st.download_button("Download bargains CSV", bargains.to_csv(index=False).encode(), "top_bargains_2023.csv","text/csv")

with col2:
    st.subheader("Top Overpays — 2023")
    tbl = filt(overpays).assign(
        pred_salary=lambda d: fmt(d['pred_salary']),
        salary_aav_usd=lambda d: fmt(d['salary_aav_usd']),
        residual=lambda d: fmt(d['residual'])
    )
    st.dataframe(tbl, use_container_width=True)
    st.download_button("Download overpays CSV", overpays.to_csv(index=False).encode(), "top_overpays_2023.csv","text/csv")

st.markdown("---")
st.caption("Model: HistGradientBoostingRegressor on log(target). Features include prior-year WAR/BB%/K%/ISO/wOBA/xwOBA, age, position, service years, prior salary, team payroll.")
