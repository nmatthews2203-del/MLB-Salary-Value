import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# =========================
# Page setup
# =========================
st.set_page_config(page_title="MLB Salary Value Explorer — 2023", page_icon="⚾", layout="wide")

TOP_N = 15                 # rows per table
LOGO_DIR = Path("logos")   # optional local logos folder
DATA_DIR = Path("outputs") # CSVs live here

# Two-way players to exclude from hitter-only rankings
TWO_WAY_PLAYERS = {"Shohei Ohtani"}  # add more if needed

# =========================
# Helpers
# =========================
def _money(x):
    try:
        return f"${int(round(float(x), 0)):,}"
    except Exception:
        return x

def _war1(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return x

def _wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = max(np.abs(y_true).sum(), 1e-9)
    return np.abs(y_true - y_pred).sum() / denom

def _style_value_colors(series: pd.Series):
    """Green for bargains (negative residuals), red for overpays (positive).
    NOTE: Value = Actual − Pred."""
    return ["color: #22c55e" if v < 0 else ("color: #ef4444" if v > 0 else "") for v in series]

def _make_styler(df_num: pd.DataFrame):
    """Pretty formatting with $, commas, and colored Value column; hide index."""
    money_cols = ["Actual Salary","Predicted Salary","Value (Actual − Pred)",
                  "Prior Salary (t−1)","Team Payroll (t−1)"]
    fmt = {c: _money for c in money_cols if c in df_num.columns}
    if "WAR (t−1)" in df_num.columns:
        fmt["WAR (t−1)"] = _war1
    if "Service Yrs (t−1)" in df_num.columns:
        fmt["Service Yrs (t−1)"] = lambda x: f"{float(x):.0f}" if pd.notna(x) else ""

    sty = (
        df_num.style
        .format(fmt)
        .set_properties(subset=["Player"], **{"font-weight": "600"})
        .hide(axis="index")
    )
    if "Value (Actual − Pred)" in df_num.columns:
        sty = sty.apply(_style_value_colors, subset=["Value (Actual − Pred)"])
    return sty

def _add_logo_column(df: pd.DataFrame) -> pd.DataFrame:
    """If a local logo file exists for Team (e.g., logos/LAD.png), add a Logo column with its path."""
    if "Team" not in df.columns:
        return df
    def logo_path(team):
        if pd.isna(team): return None
        path = LOGO_DIR / f"{str(team).strip()}.png"
        return str(path) if path.exists() else None
    out = df.copy()
    out["Logo"] = out["Team"].map(logo_path)
    if out["Logo"].notna().any():
        cols = out.columns.tolist()
        cols.insert(cols.index("Team")+1, cols.pop(cols.index("Logo")))
        out = out[cols]
    else:
        out = out.drop(columns=["Logo"])
    return out

# Column standardization so we’re robust to slight header differences
RENAME_MAP = {
    # raw -> display
    "name": "Player",
    "player": "Player",
    "team": "Team",
    "year": "Year",
    "salary_aav_usd": "Actual Salary",
    "war_tminus1": "WAR (t−1)",
    "service_years_tminus1": "Service Yrs (t−1)",
    "salary_tminus1": "Prior Salary (t−1)",
    "team_payroll_tminus1": "Team Payroll (t−1)",
    "pred_salary": "Predicted Salary",
    "residual": "Value (Actual − Pred)",
    # common alternates
    "actual_salary": "Actual Salary",
    "predicted_salary": "Predicted Salary",
    "diff": "Value (Actual − Pred)",
    "pa": "PA",
    "PA": "PA",
    "age": "Age",
    "position": "Position",
}

DISPLAY_ORDER = [
    "Player","Team","Year",
    "Actual Salary","Predicted Salary","Value (Actual − Pred)",
    "WAR (t−1)","PA","Age","Position",
    "Service Yrs (t−1)","Prior Salary (t−1)","Team Payroll (t−1)"
]

def tidy_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={k: v for k, v in RENAME_MAP.items() if k in out.columns})
    keep = [c for c in DISPLAY_ORDER if c in out.columns]
    return out[keep].copy()

@st.cache_data(show_spinner=False)
def load_outputs():
    bargains_path = DATA_DIR / "top_bargains_2023.csv"
    overpays_path = DATA_DIR / "top_overpays_2023.csv"
    if not bargains_path.exists() or not overpays_path.exists():
        st.error("Missing CSVs in outputs/. Please make sure `outputs/top_bargains_2023.csv` and `outputs/top_overpays_2023.csv` exist.")
        st.stop()
    b = pd.read_csv(bargains_path)
    o = pd.read_csv(overpays_path)
    return b, o

def compute_metrics(df: pd.DataFrame):
    if not {"Actual Salary","Predicted Salary"}.issubset(df.columns) or len(df) == 0:
        return np.nan, np.nan
    y = df["Actual Salary"].values
    yhat = df["Predicted Salary"].values
    mae = np.mean(np.abs(y - yhat))
    wape = _wape(y, yhat) * 100
    return mae, wape

def filter_and_sort(df: pd.DataFrame, team_sel, player_query, min_war, min_pa, sort_abs, is_bargain):
    out = df.copy()
    # Filters
    if team_sel:
        out = out[out["Team"].isin(team_sel)]
    if player_query:
        out = out[out["Player"].str.contains(player_query, case=False, na=False)]
    if min_war is not None and "WAR (t−1)" in out.columns:
        out = out[(out["WAR (t−1)"].astype(float) >= min_war) | out["WAR (t−1)"].isna()]
    if min_pa is not None and "PA" in out.columns:
        out = out[(out["PA"].astype(float) >= min_pa) | out["PA"].isna()]

    # Sort by value (remember: Value = Actual − Pred; negative = bargain)
    if "Value (Actual − Pred)" in out.columns:
        if sort_abs:
            out["abs_val"] = out["Value (Actual − Pred)"].abs()
            out = out.sort_values("abs_val", ascending=False).drop(columns="abs_val")
        else:
            out = out.sort_values("Value (Actual − Pred)", ascending=not is_bargain)

    out = out.head(TOP_N).copy()
    # Leaderboard rank
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out

# =========================
# Load & prep data
# =========================
bargains_raw, overpays_raw = load_outputs()

# Robust column handling -> standardized display schema
bargains = tidy_cols(bargains_raw)
overpays  = tidy_cols(overpays_raw)

# Exclude two-way players
if "Player" in bargains.columns:
    bargains = bargains[~bargains["Player"].isin(TWO_WAY_PLAYERS)]
if "Player" in overpays.columns:
    overpays = overpays[~overpays["Player"].isin(TWO_WAY_PLAYERS)]

# =========================
# Sidebar (FORM with clear-on-submit)
# =========================
st.sidebar.header("Filters")

show_war = ("WAR (t−1)" in bargains.columns) or ("WAR (t−1)" in overpays.columns)
show_pa  = ("PA" in bargains.columns) or ("PA" in overpays.columns)
teams = sorted(set(bargains["Team"].dropna().unique()).union(set(overpays["Team"].dropna().unique()))) if "Team" in bargains.columns and "Team" in overpays.columns else []

with st.sidebar.form("filters", clear_on_submit=True):
    team_sel = st.multiselect("Teams", teams, default=[])
    player_query = st.text_input("Search player", value="", placeholder="e.g., Alonso")

    min_war = st.number_input("Min WAR (t−1, optional)", value=0.0, min_value=0.0, step=0.5) if show_war else None
    min_pa  = st.number_input("Min PA (optional)", value=0.0, min_value=0.0, step=25.0) if show_pa else None

    sort_abs = st.toggle("Sort by absolute value gap (largest first)", value=False)
    metrics_follow_filters = st.toggle("Metrics reflect filters", value=True)

    c1, c2 = st.columns(2)
    apply_clicked = c1.form_submit_button("Apply filters", use_container_width=True)
    clear_clicked = c2.form_submit_button("Clear", type="secondary", use_container_width=True)

# If user hit Clear, use defaults for this run (widgets already reset by clear_on_submit)
if clear_clicked:
    team_sel = []
    player_query = ""
    if show_war: min_war = 0.0
    if show_pa:  min_pa = 0.0
    sort_abs = False
    metrics_follow_filters = True

# Optional logo (show if exactly one team selected and we have a logo file)
if team_sel and len(team_sel) == 1:
    logo_path = LOGO_DIR / f"{team_sel[0]}.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Two-way players (e.g., Shohei Ohtani) are excluded from rankings since pitching isn’t modeled.")

# =========================
# Header
# =========================
st.title("⚾ MLB Salary Value Explorer — 2023")
st.caption(
    "Predicts year-t salary from t−1 performance & context to surface **bargains** and **overpays**. "
    "Model: HistGradientBoostingRegressor on log-salary; train ≤ 2022, test = 2023."
)
st.markdown("---")

# =========================
# Build filtered views once (reuse for metrics + tabs)
# =========================
b_view = filter_and_sort(bargains, team_sel, player_query, min_war, min_pa, sort_abs, is_bargain=True)
o_view = filter_and_sort(overpays, team_sel, player_query, min_war, min_pa, sort_abs, is_bargain=False)

# Choose which dataset metrics use
both_unfiltered = pd.concat([bargains, overpays], ignore_index=True)
if metrics_follow_filters:
    metrics_df = pd.concat([b_view.drop(columns=["Rank"], errors="ignore"),
                            o_view.drop(columns=["Rank"], errors="ignore")], ignore_index=True)
    label = "view"
else:
    metrics_df = both_unfiltered
    label = "2023"

mae_all, wape_all = compute_metrics(metrics_df)

mc1, mc2 = st.columns(2)
with mc1:
    st.metric(f"MAE ({label})", _money(mae_all) if np.isfinite(mae_all) else "—")
with mc2:
    st.metric(f"WAPE ({label})", f"{wape_all:.1f}%" if np.isfinite(wape_all) else "—")

# =========================
# Tables
# =========================
st.subheader("Rankings")
t1, t2 = st.tabs(["✅ Top Bargains", "⚠️ Top Overpays"])

with t1:
    st.dataframe(_make_styler(_add_logo_column(b_view)), use_container_width=True)
    # Download (source, two-way filtered only)
    if "name" in bargains_raw.columns:
        b_dl = bargains_raw[~bargains_raw["name"].isin(TWO_WAY_PLAYERS)]
    elif "player" in bargains_raw.columns:
        b_dl = bargains_raw[~bargains_raw["player"].isin(TWO_WAY_PLAYERS)]
    else:
        b_dl = bargains_raw
    st.download_button(
        "Download bargains (source, two-way filtered)",
        b_dl.to_csv(index=False).encode("utf-8"),
        "top_bargains_2023.csv",
        "text/csv"
    )

with t2:
    st.dataframe(_make_styler(_add_logo_column(o_view)), use_container_width=True)
    if "name" in overpays_raw.columns:
        o_dl = overpays_raw[~overpays_raw["name"].isin(TWO_WAY_PLAYERS)]
    elif "player" in overpays_raw.columns:
        o_dl = overpays_raw[~overpays_raw["player"].isin(TWO_WAY_PLAYERS)]
    else:
        o_dl = overpays_raw
    st.download_button(
        "Download overpays (source, two-way filtered)",
        o_dl.to_csv(index=False).encode("utf-8"),
        "top_overpays_2023.csv",
        "text/csv"
    )

# Combined download of the *filtered views* (what the user is currently seeing)
st.markdown("### Download current view (filtered)")
comb = b_view.assign(category="Bargain").pipe(lambda d: pd.concat([d, o_view.assign(category="Overpay")], ignore_index=True))
st.download_button(
    "Download combined (filtered view)",
    comb.to_csv(index=False).encode("utf-8"),
    "mlb_value_filtered_combined.csv",
    "text/csv"
)

# Player spotlight
st.markdown("---")
st.subheader("Player spotlight")
spot_df = comb.copy()
player_options = spot_df["Player"].dropna().unique().tolist() if "Player" in spot_df.columns else []
if player_options:
    pick = st.selectbox("Choose a player", ["—"] + player_options, index=0)
    if pick != "—":
        card = spot_df[spot_df["Player"] == pick].head(1).copy()
        st.dataframe(_make_styler(_add_logo_column(card)), use_container_width=True)
else:
    st.caption("No players available in the current filtered view.")

# =========================
# About / Caveats
# =========================
with st.expander("About this app & caveats"):
    st.markdown("""
- **Goal:** Surface 2023 hitter salary inefficiencies by predicting salaries from prior-year performance & context.
- **Model:** HistGradientBoostingRegressor on **log(salary)**; features include prior-year WAR/plate discipline, age, position group, service time, prior salary, team payroll. Train ≤ 2022, Test = 2023.
- **Value column:** `Actual − Pred`. Negative → **bargain** (model thinks salary should be higher). Positive → **overpay**.
- **Exclusions:** Two-way players (e.g., Shohei Ohtani) are filtered out since pitching isn’t modeled.
- **Caveats:** Arbitration rules, multi-year deals, injuries, and option/bonus structures aren’t fully captured; use directionally.
""")
