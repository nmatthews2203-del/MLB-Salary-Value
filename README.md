# MLB Salary Value Explorer

**Live demo:** [Click here to view the Streamlit app](https://mlb-salary-value-mnzenriikbeqp4lmqqtdnz.streamlit.app/)

Predicts MLB **year _t_ salary** from **year _t−1_ performance + context** and highlights **2023 bargains and overpays**.


---

## Project Overview

This project uses baseball stats and salary data to model and evaluate MLB player salaries. It’s designed to showcase skills in **data collection**, **feature engineering**, **machine learning**, and **interactive dashboards**.

**Data Sources:**
- **Stats:** pybaseball (2018–2024)
- **Salaries:** Kaggle MLB salaries dataset (2011–2024)

**Model:**
- **HistGradientBoostingRegressor** on log-transformed salary
- **Time-aware split:** trained on seasons ≤2022, tested on 2023

**Performance (2023):**
- **MAE:** ≈ $5.7M  
- **WAPE:** ≈ 45%

**Outputs:**
- `top_bargains_2023.csv`
- `top_overpays_2023.csv`

---

## How to Run

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv && source .venv/bin/activate