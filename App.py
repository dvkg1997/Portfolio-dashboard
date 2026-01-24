import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io
from scipy.optimize import minimize
from fpdf import FPDF

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Institutional Portfolio Engine")

INDEX_MAP = {
    # --- Broad Market Indices ---
    "Nifty 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "Nifty Next 50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "Nifty 100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "Nifty Midcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
    "Nifty 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "Nifty Microcap 250": "https://www.niftyindices.com/IndexConstituent/ind_nifty_microcap250_list.csv",

    # --- Sectoral Indices ---
    "Nifty Auto": "https://www.niftyindices.com/IndexConstituent/ind_niftyautolist.csv",
    "Nifty Bank": "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv",
    "Nifty Consumer Durables": "https://www.niftyindices.com/IndexConstituent/ind_niftyconsumerdurableslist.csv",
    "Nifty Financial Services": "https://www.niftyindices.com/IndexConstituent/ind_niftyfinancelist.csv",
    "Nifty FMCG": "https://www.niftyindices.com/IndexConstituent/ind_niftyfmcglist.csv",
    "Nifty Healthcare": "https://www.niftyindices.com/IndexConstituent/ind_niftyhealthcarelist.csv",
    "Nifty IT": "https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv",
    "Nifty Media": "https://www.niftyindices.com/IndexConstituent/ind_niftymedialist.csv",
    "Nifty Metal": "https://www.niftyindices.com/IndexConstituent/ind_niftymetallist.csv",
    "Nifty Oil & Gas": "https://www.niftyindices.com/IndexConstituent/ind_niftyoilgaslist.csv",
    "Nifty Pharma": "https://www.niftyindices.com/IndexConstituent/ind_niftypharmalist.csv",
    "Nifty Realty": "https://www.niftyindices.com/IndexConstituent/ind_niftyrealtylist.csv",

    # --- Thematic Indices ---
    "Nifty Commodities": "https://www.niftyindices.com/IndexConstituent/ind_niftycommoditieslist.csv",
    "Nifty Energy": "https://www.niftyindices.com/IndexConstituent/ind_niftyenergylist.csv",
    "Nifty India Consumption": "https://www.niftyindices.com/IndexConstituent/ind_niftyconsumptionlist.csv",
    "Nifty Infrastructure": "https://www.niftyindices.com/IndexConstituent/ind_niftyinfralist.csv",
    "Nifty MNC": "https://www.niftyindices.com/IndexConstituent/ind_niftymnclist.csv",
    "Nifty PSE": "https://www.niftyindices.com/IndexConstituent/ind_niftypselist.csv",
    "Nifty Services Sector": "https://www.niftyindices.com/IndexConstituent/ind_niftyservicelist.csv",

    # --- Strategy Indices ---
    "Nifty 50 Value 20": "https://www.niftyindices.com/IndexConstituent/ind_nifty50Value20list.csv",
    "Nifty Alpha 50": "https://www.niftyindices.com/IndexConstituent/ind_niftyalpha50list.csv",
    "Nifty High Beta 50": "https://www.niftyindices.com/IndexConstituent/ind_niftyhighbeta50list.csv",
    "Nifty Low Volatility 50": "https://www.niftyindices.com/IndexConstituent/ind_niftylowvolatility50list.csv"
}
HAVEN_MAP = {
    "Gold (India)": "GOLDBEES.NS",
    "Indian G-Sec (10Y)": "NIFTYGS10.NS",
    "Gold (Global)": "GC=F",
    "US Treasury (10Y)": "TLT"
}
# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Parameters")
    selected_index = st.selectbox("Universe", list(INDEX_MAP.keys()))
    strategy = st.radio("Goal", ["Max Sharpe", "Min Volatility", "Aggressive"])

    st.divider()
    samp_size = st.slider("Stock Sample Size", 10, 60, 30)
    w_cap = st.slider("Max Weight per Stock (%)", 5, 50, 15) / 100
    rf_rate = st.number_input("Risk-Free Rate (%)", value=7.0) / 100

    st.header("💎 Safe Havens")
    havens = st.multiselect("Add Hedges", list(HAVEN_MAP.keys()), default=["Gold (India)"])

    st.header("💥 Stress Testing")
    stress_k = st.slider("Systemic Stress (k)", 0.0, 1.0, 0.0, 0.05)
    d_alpha = st.select_slider("Dirichlet Alpha (Dispersion)", options=[0.01, 0.05, 0.1, 1.0], value=0.05)

    run_btn = st.button("🚀 EXECUTE ANALYSIS", use_container_width=True)


# --- FUNCTIONS ---
def get_stats(w, mean_ret, cov_mat, rf):
    p_ret = np.dot(w, mean_ret)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0
    return p_ret, p_vol, sharpe


def create_pdf(kpis, weights, selected_index):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Portfolio Fact Sheet: {selected_index}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Executive KPIs", ln=True)
    pdf.set_font("Arial", '', 10)
    for k, v in kpis.items(): pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Top 15 Holdings", ln=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(90, 8, "Ticker", 1)
    pdf.cell(90, 8, "Weight (%)", 1, ln=True)
    pdf.set_font("Arial", '', 10)
    for _, row in weights.head(15).iterrows():
        pdf.cell(90, 8, str(row['Ticker']), 1)
        pdf.cell(90, 8, f"{row['Weight']:.2f}%", 1, ln=True)
    return pdf.output(dest='S').encode('latin-1')


# --- MAIN LOGIC ---
if run_btn:
    # A. Fetch Data
    headers = {'User-Agent': 'Mozilla/5.0'}
    index_df = pd.read_csv(io.StringIO(requests.get(INDEX_MAP[selected_index], headers=headers).text))
    symbols = [s.strip() + ".NS" for s in index_df.sample(min(samp_size, len(index_df)))['Symbol'].tolist()]
    haven_tickers = [HAVEN_MAP[h] for h in havens]
    all_tickers = list(set(symbols + haven_tickers + ["USDINR=X"]))

    with st.spinner("Crunching Financial Data..."):
        raw_data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)['Close']
        raw_data = raw_data.ffill().bfill()

        # B. Currency Adjust (USD to INR)
        usd_inr_rets = raw_data["USDINR=X"].pct_change().fillna(0)
        global_tickers = [HAVEN_MAP[h] for h in ["Gold (Global)", "US Treasury (10Y)"] if h in havens]

        data = raw_data.drop(columns=["USDINR=X"])
        for t in global_tickers:
            if t in data.columns:
                # Correct adjustment: (1+r_asset)*(1+r_fx)-1
                inr_rets = (1 + data[t].pct_change().fillna(0)) * (1 + usd_inr_rets) - 1
                data[t] = (1 + inr_rets).cumprod()

        returns_df = data.pct_change().dropna()
        mean_ret = returns_df.mean() * 252
        cov_mat = returns_df.cov() * 252

        # C. Systemic Stress (Correlation Breakdown)
        vols = np.sqrt(np.diag(cov_mat))
        corr_mat = returns_df.corr()
        stressed_corr = (1 - stress_k) * corr_mat + stress_k * np.ones_like(corr_mat)
        cov_mat = np.outer(vols, vols) * stressed_corr

        # D. Optimize Main Portfolio
        num_assets = len(mean_ret)


        def objective(w):
            r, v, s = get_stats(w, mean_ret, cov_mat, rf_rate)
            if strategy == "Min Volatility": return v
            if strategy == "Aggressive": return -r
            return -s


        res = minimize(objective, num_assets * [1. / num_assets],
                       bounds=tuple((0, w_cap) for _ in range(num_assets)),
                       constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        opt_ret, opt_vol, opt_sharpe = get_stats(res.x, mean_ret, cov_mat, rf_rate)

        # E. Efficient Frontier Boundary (The Black Dots)
        # Sweep from Min-Variance Return to Max Individual Asset Return
        target_rets = np.linspace(mean_ret.min(), mean_ret.max(), 30)
        frontier_vols = []
        valid_rets = []
        for tr in target_rets:
            c = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                 {'type': 'eq', 'fun': lambda x: np.dot(x, mean_ret) - tr})
            eff = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_mat, w))),
                           num_assets * [1. / num_assets], bounds=tuple((0, w_cap) for _ in range(num_assets)),
                           constraints=c)
            if eff.success:
                frontier_vols.append(eff.fun)
                valid_rets.append(tr)

        # F. Cholesky Stress Test (CVaR)
        L = np.linalg.cholesky(cov_mat + np.eye(num_assets) * 1e-9)
        sim_shocks = L @ np.random.normal(size=(num_assets, 10000))
        # Portfolio simulated daily returns
        sim_p_rets = np.dot(res.x, sim_shocks)
        cvar_99 = sim_p_rets[sim_p_rets <= np.percentile(sim_p_rets, 1)].mean()

    # --- DASHBOARD UI ---
    st.header(f"💼 Analysis: {selected_index} Optimized")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected Return", f"{opt_ret * 100:.2f}%")
    k2.metric("Portfolio Risk", f"{opt_vol * 100:.2f}%")
    k3.metric("Sharpe Ratio", round(opt_sharpe, 2))
    k4.metric("99% Daily CVaR", f"{cvar_99 * 100:.2f}%")

    # Efficient Frontier Chart
    # Use Dot Product for simulated returns
    w_mc = np.random.dirichlet(np.ones(num_assets) * d_alpha, 10000)
    mc_rets = w_mc @ mean_ret
    # Vectorized Volatility calculation
    mc_vols = np.sqrt(np.einsum('ij,jk,ik->i', w_mc, cov_mat, w_mc))
    mc_sharpe = (mc_rets - rf_rate) / mc_vols

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_vols, y=mc_rets, mode='markers',
                             marker=dict(color=mc_sharpe, colorscale='Viridis', size=3, opacity=0.4), name="Simulated"))
    fig.add_trace(go.Scatter(x=frontier_vols, y=valid_rets, mode='lines',
                             line=dict(color='black', width=2, dash='dash'), name="Efficient Frontier"))
    fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers',
                             marker=dict(color='red', symbol='star', size=15), name=f"Target: {strategy}"))
    fig.update_layout(xaxis_title="Volatility (Risk)", yaxis_title="Annualized Return", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Tables and Heatmap
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Optimal Weights")
        w_df = pd.DataFrame({'Ticker': data.columns, 'Weight': res.x * 100}).sort_values('Weight', ascending=False)
        st.dataframe(w_df.style.format({"Weight": "{:.2f}%"}), height=450)

    with c2:
        st.subheader("Correlation Matrix (Stressed)")
        fig_h, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(stressed_corr, columns=data.columns, index=data.columns), cmap="RdYlBu", ax=ax)
        st.pyplot(fig_h)

    # Export
    pdf_bytes = create_pdf(
        {"Ann. Return": f"{opt_ret * 100:.2f}%", "Risk": f"{opt_vol * 100:.2f}%", "Sharpe": opt_sharpe,
         "CVaR": f"{cvar_99 * 100:.2f}%"}, w_df, selected_index)
    st.download_button("📥 Download PDF Report", pdf_bytes, "Investment_Fact_Sheet.pdf", "application/pdf",
                       use_container_width=True)
