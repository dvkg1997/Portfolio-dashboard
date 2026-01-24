import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io
import os
import webbrowser
from scipy.optimize import minimize
from fpdf import FPDF
import datetime

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(layout="wide", page_title="Institutional Portfolio Engine")

# Extended NSE Index Mapping
INDEX_MAP = {
    "Nifty 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "Nifty Next 50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "Nifty 100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "Nifty Midcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "Nifty Bank": "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv",
    "Nifty IT": "https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv",
    "Nifty Oil & Gas": "https://www.niftyindices.com/IndexConstituent/ind_niftyoilgaslist.csv"
}

HAVEN_MAP = {
    "Gold (India)": "GOLDBEES.NS",
    "Indian G-Sec (10Y)": "NIFTYGS10.NS",
    "Gold (Global)": "GC=F",
    "US Treasury (10Y)": "TLT"
}

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🎯 Parameters")
    selected_index = st.selectbox("Universe", list(INDEX_MAP.keys()))
    strategy = st.radio("Goal", ["Max Sharpe", "Min Volatility", "Aggressive"])

    st.divider()
    samp_size = st.slider("Sample Size", 10, 60, 30)
    w_cap = st.slider("Max Weight per Stock (%)", 5, 50, 15) / 100
    rf_rate = st.number_input("Risk-Free Rate (%)", value=7.0) / 100

    st.divider()
    st.header("💎 Safe Havens")
    havens = st.multiselect("Add Hedges", list(HAVEN_MAP.keys()), default=["Gold (India)"])

    st.header("💥 Stress Test")
    stress_k = st.slider("Systemic Stress (k)", 0.0, 1.0, 0.0, 0.05, help="Blends correlation toward 1.0")
    d_alpha = st.select_slider("Dirichlet Alpha", options=[0.01, 0.05, 0.1, 1.0], value=0.05)

    run_btn = st.button("🚀 EXECUTE ANALYSIS", use_container_width=True)


# --- 3. HELPER FUNCTIONS ---
def get_stats(w,mean_ret, cov_mat, rf):
    p_ret = np.sum(mean_ret * w)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0
    return p_ret, p_vol, sharpe


def create_pdf(kpis, weights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Investment Fact Sheet", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Portfolio KPIs", ln=True)
    pdf.set_font("Arial", '', 10)
    for k, v in kpis.items(): pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top Allocations", ln=True)
    pdf.set_font("Arial", '', 10)
    for i, row in weights.head(10).iterrows():
        pdf.cell(100, 8, str(row['Ticker']), 1)
        pdf.cell(80, 8, f"{row['Weight']:.2f}%", 1, ln=True)
    return pdf.output(dest='S').encode('latin-1')


# --- 4. MAIN EXECUTION ---
if run_btn:
    # A. Fetch Assets
    headers = {'User-Agent': 'Mozilla/5.0'}
    index_df = pd.read_csv(io.StringIO(requests.get(INDEX_MAP[selected_index], headers=headers).text))
    symbols = [s.strip() + ".NS" for s in index_df.sample(samp_size)['Symbol'].tolist()]
    haven_tickers = [HAVEN_MAP[h] for h in havens]
    all_tickers = symbols + haven_tickers + ["USDINR=X"]

    with st.spinner("Processing Market Data..."):
        raw_data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)['Close']
        raw_data = raw_data.ffill().bfill()

        # B. Currency Adjustment (USD to INR)
        usd_inr_rets = raw_data["USDINR=X"].pct_change().fillna(0)
        global_tickers = [HAVEN_MAP[h] for h in ["Gold (Global)", "US Treasury (10Y)"] if h in havens]

        data = raw_data.drop(columns=["USDINR=X"])
        for t in global_tickers:
            if t in data.columns:
                inr_rets = (1 + data[t].pct_change().fillna(0)) * (1 + usd_inr_rets) - 1
                data[t] = (1 + inr_rets).cumprod()

        returns = data.pct_change().dropna()
        mean_ret = returns.mean() * 252
        cov_mat = returns.cov() * 252

        # C. Apply Systemic Stress (k)
        vols = np.sqrt(np.diag(cov_mat))
        corr = returns.corr()
        stressed_corr = (1 - stress_k) * corr + stress_k * np.ones_like(corr)
        cov_mat = np.outer(vols, vols) * stressed_corr

        # D. Optimize
        num_assets = len(mean_ret)


        def obj(w):
            r, v, s = get_stats(w, mean_ret, cov_mat, rf_rate)
            return v if strategy == "Min Volatility" else (-r if strategy == "Aggressive" else -s)


        res = minimize(obj, num_assets * [1. / num_assets], bounds=tuple((0, w_cap) for _ in range(num_assets)),
                       constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        opt_ret, opt_vol, opt_sharpe = get_stats(res.x, mean_ret, cov_mat, rf_rate)

        # E. Cholesky Stress Test
        L = np.linalg.cholesky(cov_mat + np.eye(num_assets) * 1e-9)
        sim_rets = np.dot(res.x, L @ np.random.normal(size=(num_assets, 10000)))
        cvar_99 = sim_rets[sim_rets <= np.percentile(sim_rets, 1)].mean()

    # --- 5. VISUALIZATION ---
    st.title(f"📊 {selected_index} + Hedges")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ann. Return", f"{opt_ret * 100:.2f}%")
    k2.metric("Portfolio Risk", f"{opt_vol * 100:.2f}%")
    k3.metric("Sharpe Ratio", round(opt_sharpe, 2))
    k4.metric("99% CVaR", f"{cvar_99 * 100:.2f}%")

    # Efficient Frontier
    w_mc = np.random.dirichlet(np.ones(num_assets) * d_alpha, 10000)
    mc_rets = np.sum(mean_ret * w_mc, axis=1)
    mc_vols = np.array([np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) for w in w_mc])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_vols, y=mc_rets, mode='markers',
                             marker=dict(color=(mc_rets - rf_rate) / mc_vols, colorscale='Viridis', size=3,
                                         opacity=0.4), name="Simulated"))
    fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', marker=dict(color='red', symbol='star', size=15),
                             name="Optimum"))
    st.plotly_chart(fig, use_container_width=True)

    # Weights Table
    w_df = pd.DataFrame({'Ticker': data.columns, 'Weight': res.x * 100}).sort_values('Weight', ascending=False)
    st.subheader("Asset Allocation")
    st.dataframe(w_df.style.format({"Weight": "{:.2f}%"}), use_container_width=True)

    # PDF Download
    pdf_b = create_pdf({"Return": f"{opt_ret * 100:.2f}%", "Risk": f"{opt_vol * 100:.2f}%", "Sharpe": opt_sharpe,
                        "CVaR": f"{cvar_99 * 100:.2f}%"}, w_df)
    st.download_button("📥 Download PDF Fact Sheet", pdf_b, "Report.pdf", "application/pdf")