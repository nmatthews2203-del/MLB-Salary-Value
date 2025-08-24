import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="MLB Salary Value Explorer — 2023", layout="wide")

TOP_N = 15               # fixed number of rows per table
LOGO_DIR = Path("logos") # optional local logos folder

# Two-way players to exclude from hitter-only rankings
TWO_WAY_PLAYERS = {"Shohei Ohtani"}  # add more if needed

# ---------- helpers ----------
@st.cache_data
def load_outputs():
    bargains = pd.read_csv("top_bargains_2023.csv")
    overpays = pd.read_csv("top_overpays_2023.csv")
    return bargains, overpays

def tidy_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for display / modeling."""
    rename_map = {
        "name": "Player",
        "team": "Team",
        "year": "Year",
        "salary_aav_usd": "Actual Salary",
        "war_tminus1": "WAR (t−1)",
        "service_years_tminus1": "Service Yrs (t−1)",
        "salary_tminus1": "Prior Salary (t−1)",
        "team_payroll_tminus1": "Team Payroll (t−1)",
        "pred_salary": "Predicted Salary",
        "residual": "Value (Actual − Pred)",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    order = [
        "Player","Team","Year","Actual Salary","Predicted Salary","Value (Actual − Pred)",
        "WAR (t−1)","Service Yrs (t−1)","Prior Salary (t−1)","Team Payroll (t−1)"
    ]
    return df[[c for c in order if c in df.columns]]

def compute_metrics(df: pd.DataFrame):
    if not {"Actual Salary","Predicted Salary"}.issubset(df.columns) or len(df) == 0:
        return np.nan, np.nan
    y = df["Actual Salary"].values
    yhat = df["Predicted Salary"].values
    mae = np.mean(np.abs(y - yhat))
    wape = np.sum(np.abs(y - yhat)) / np.sum(np.abs(y)) * 100
    return mae, wape

def _style_value_colors(series: pd.Series):
    """Green for bargains (negative residuals), red for overpays (positive)."""
    return [
        "color: #22c55e" if v < 0 else ("color: #ef4444" if v > 0 else "")
        for v in series
    ]

def make_styler(df_num: pd.DataFrame):
    """Pretty formatting with $, commas, and colored Value column."""
    money_cols = ["Actual Salary","Predicted Salary","Value (Actual − Pred)",
                  "Prior Salary (t−1)","Team Payroll (t−1)"]
    fmt = {c: "${:,.0f}" for c in money_cols if c in df_num.columns}
    if "WAR (t−1)" in df_num.columns:
        fmt["WAR (t−1)"] = "{:.1f}"
    if "Service Yrs (t−1)" in df_num.columns:
        fmt["Service Yrs (t−1)"] = "{:.0f}"

    sty = (
        df_num.style
        .format(fmt)
        .set_properties(subset=["Player"], **{"font-weight": "600"})
    )
    if "Value (Actual − Pred)" in df_num.columns:
        sty = sty.apply(_style_value_colors, subset=["Value (Actual − Pred)"])
    return sty

def add_logo_column(df: pd.DataFrame) -> pd.DataFrame:
    """If a local logo file exists for Team (e.g., logos/LAD.png), add a Logo column with its path."""
    if "Team" not in df.columns:
        return df
    def logo_path(team):
        if pd.isna(team): return None
        path = LOGO_DIR / f"{str(team).strip()}.png"
        return str(path) if path.exists() else None
    out = df.copy()
    out["Logo"] = out["Team"].map(logo_path)
    # Put Logo next to Team if any logos exist
    if out["Logo"].notna().any():
        cols = out.columns.tolist()
        if "Logo" in cols and "Team" in cols:
            cols.insert(cols.index("Team")+1, cols.pop(cols.index("Logo")))
            out = out[cols]
    else:
        out = out.drop(columns=["Logo"])
    return out

# ---------- UI ----------
st.title("MLB Salary Value Explorer — 2023")
st.markdown(
    "_Predicts year-t salary from t−1 performance + context. "
    "Use the team filter on the left to explore. Data: pybaseball (stats) & Kaggle (salaries)._"
)
st.markdown("---")

# Sidebar
st.sidebar.subheader("Filter by team")
bargains_raw, overpays_raw = load_outputs()
teams = sorted(set(bargains_raw["team"].dropna()) | set(overpays_raw["team"].dropna()))
team_sel = st.sidebar.multiselect("Teams", teams, default=[])

# Prepare numeric tables
bargains = tidy_cols(bargains_raw)
overpays  = tidy_cols(overpays_raw)

# Filter by team
if team_sel:
    bargains = bargains[bargains["Team"].isin(team_sel)]
    overpays = overpays[overpays["Team"].isin(team_sel)]

# ---- Exclude two-way players from hitter-only rankings (e.g., Shohei Ohtani) ----
bargains = bargains[~bargains["Player"].isin(TWO_WAY_PLAYERS)]
overpays = overpays[~overpays["Player"].isin(TWO_WAY_PLAYERS)]

# Sort & limit: most negative residuals are biggest bargains
if "Value (Actual − Pred)" in bargains.columns:
    bargains = bargains.sort_values("Value (Actual − Pred)").head(TOP_N)
if "Value (Actual − Pred)" in overpays.columns:
    overpays = overpays.sort_values("Value (Actual − Pred)", ascending=False).head(TOP_N)

# Metrics (on the shown rows for quick context)
mae, wape = compute_metrics(pd.concat([bargains, overpays], ignore_index=True))

# Badges row
b1, b2 = st.columns(2)
with b1:
    st.markdown("**Most undervalued (by model)**")
with b2:
    st.markdown("**Most overpaid (by model)**")

# Metrics row
m1, m2 = st.columns(2)
with m1:
    st.metric("MAE (shown)", f"${mae:,.0f}" if np.isfinite(mae) else "—")
with m2:
    st.metric("WAPE (shown)", f"{wape:.1f}%" if np.isfinite(wape) else "—")

st.markdown("---")

# Tables (always pretty-formatted + value coloring + optional logos)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Bargains — 2023")
    st.dataframe(make_styler(add_logo_column(bargains)), use_container_width=True)

    # Match downloads to the same two-way filter
    bargains_dl = bargains_raw[~bargains_raw["name"].isin(TWO_WAY_PLAYERS)]
    st.download_button(
        "Download bargains CSV",
        bargains_dl.to_csv(index=False).encode("utf-8"),
        "top_bargains_2023.csv",
        "text/csv"
    )

with col2:
    st.subheader("Top Overpays — 2023")
    st.dataframe(make_styler(add_logo_column(overpays)), use_container_width=True)

    overpays_dl = overpays_raw[~overpays_raw["name"].isin(TWO_WAY_PLAYERS)]
    st.download_button(
        "Download overpays CSV",
        overpays_dl.to_csv(index=False).encode("utf-8"),
        "top_overpays_2023.csv",
        "text/csv"
    )

st.markdown("---")
st.caption(
    "Model uses hitter-only features. Two-way players (e.g., Shohei Ohtani) are excluded from rankings "
    "because pitching value is not modeled."
)
