import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from arch import arch_model
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde, norm
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import requests
from datetime import datetime, timezone, timedelta
import re
import logging
from typing import Optional, Dict, List
import time
from statsmodels.tsa.stattools import adfuller

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
st.set_page_config(layout="wide", page_icon="📊", page_title="BTC Options and Stochastic Dynamics Analysis")
st.title("BTC/USD Options and Non-Equilibrium Stochastic Dynamics Analysis")
st.markdown("""
This application combines options-based probability analysis with a non-equilibrium stochastic model for BTC/USD, based on a generalized Hull-White model with arbitrage return.
- **Options Analysis**: Implied volatility (IV) smiles and probability density functions (PDFs) using SVI model, ensuring no butterfly arbitrage. The PDF is used to calculate a central probability range.
- **Stochastic Dynamics**: Models price, volatility, and arbitrage return using SDEs with non-equilibrium effects (crash phases when arbitrage exceeds threshold).
- **IV/RV & Momentum**: Compares implied vs. realized volatility and includes RSI-based momentum signals.
- **Volume Profile**: Historical trading activity with S/R levels and POC.
*Use the sidebar to adjust parameters. Hover over charts for details.*
""")

# --- Thalex API Configuration ---
THALEX_BASE_URL = "https://thalex.com/api/v2"
REQUEST_TIMEOUT = 15
Z_SCORE_HIGH_IV_THRESHOLD = 1.0
Z_SCORE_LOW_IV_THRESHOLD = -1.0
RSI_BULLISH_THRESHOLD = 65
RSI_BEARISH_THRESHOLD = 35

# --- Stochastic Dynamics Simulation ---
def simulate_non_equilibrium(S0, V0, eta0, mu, phi, epsilon, lambda_, chi, alpha, eta_star, S_u, S_l, kappa, rho_XY, rho_XZ, rho_YZ, T, N, n_paths=1):
    dt = T / N
    S = np.zeros((n_paths, N+1))
    V = np.zeros((n_paths, N+1))
    eta = np.zeros((n_paths, N+1))
    S[:, 0] = S0
    V[:, 0] = V0
    eta[:, 0] = eta0

    # Correlation matrix
    corr_matrix = np.array([[1.0, rho_XY, rho_XZ],
                            [rho_XY, 1.0, rho_YZ],
                            [rho_XZ, rho_YZ, 1.0]])
    L = np.linalg.cholesky(corr_matrix)

    for t in range(N):
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, 3))
        dW_correlated = dW @ L.T

        S_t = S[:, t]
        V_t = V[:, t]
        eta_t = eta[:, t]
        eta_ratio = eta_t / eta_star
        exp_eta = np.exp(-(eta_ratio**2))
        price_bound_term = (2 * S_t / S_u - S_l / S_u - 1)**2 / (1 - (2 * S_t / S_u - S_l / S_u - 1)**2 + 1e-6)
        exp_bound = 1 - np.exp(-price_bound_term)

        # SDEs (equation 86)
        dS = mu * (1 - alpha * eta_ratio) * S_t * dt + np.sqrt(V_t) * S_t * dW_correlated[:, 0]
        dV = phi * (V0 * exp_eta - V_t) * dt + epsilon * exp_eta * np.sqrt(V_t) * dW_correlated[:, 1]
        d_eta = (-lambda_ * eta_t * exp_eta + kappa * exp_bound) * dt + chi * dW_correlated[:, 2]

        S[:, t+1] = S_t + dS
        V[:, t+1] = np.maximum(V_t + dV, 1e-6)  # Ensure non-negative volatility
        eta[:, t+1] = eta_t + d_eta

    return S, V, eta

# --- Helper Functions (Reusing Existing Ones) ---
# [Retain all helper functions: fetch_kraken_data, get_hit_prob, create_interactive_density_chart,
# create_volume_profile_chart, calculate_probability_range, get_thalex_instruments,
# get_thalex_ticker, get_thalex_options_data, simulate_options_data,
# get_expiries_from_instruments, BlackScholes, SVI functions, calculate_rsi,
# calculate_realized_volatility]

# --- Sidebar Configuration ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 90, 30)
epsilon_factor = st.sidebar.slider("S/R Zone Width Factor", 0.1, 2.0, 0.5, step=0.05)
st.session_state['profitability_threshold'] = st.sidebar.slider("Profitability Confidence Interval (%)", 68, 99, 68, step=1) / 100.0
st.sidebar.header("Stochastic Dynamics Parameters")
mu = st.sidebar.slider("Price Drift (μ)", 0.0, 0.2, 0.05, step=0.01)
phi = st.sidebar.slider("Volatility Mean Reversion (φ)", 0.1, 2.0, 1.0, step=0.1)
epsilon = st.sidebar.slider("Volatility of Volatility (ε)", 0.05, 0.5, 0.2, step=0.01)
lambda_ = st.sidebar.slider("Arbitrage Mean Reversion (λ)", 0.1, 2.0, 1.0, step=0.1)
chi = st.sidebar.slider("Arbitrage Volatility (χ)", 0.01, 0.1, 0.05, step=0.01)
alpha = st.sidebar.slider("Mispricing Impact (α)", 0.1, 1.0, 0.5, step=0.1)
eta_star = st.sidebar.slider("Mispricing Threshold (η*)", 0.01, 0.2, 0.09, step=0.01)
S_u = st.sidebar.slider("Upper Price Bound (S_u)", 80000, 120000, 110000, step=1000)
S_l = st.sidebar.slider("Lower Price Bound (S_l)", 50000, 90000, 90000, step=1000)
kappa = st.sidebar.slider("Arbitrage Revival (κ)", 0.01, 0.5, 0.1, step=0.01)
rho_XY = st.sidebar.slider("Price-Volatility Correlation (ρ_XY)", -0.5, 0.5, 0.3, step=0.05)
rho_XZ = st.sidebar.slider("Price-Arbitrage Correlation (ρ_XZ)", -0.5, 0.5, 0.2, step=0.05)
rho_YZ = st.sidebar.slider("Volatility-Arbitrage Correlation (ρ_YZ)", -0.5, 0.5, 0.0, step=0.05)
st.sidebar.header("Options Analysis Parameters")
all_instruments = get_thalex_instruments()
coin = "BTC"
expiries = get_expiries_from_instruments(all_instruments, coin)
sel_expiry = st.sidebar.selectbox("Options Expiry", expiries, index=0) if expiries else None
r_rate = st.sidebar.slider("Prime Rate (%)", 0.0, 14.0, 1.6, 0.1) / 100.0
use_oi_weights = st.sidebar.checkbox("Use OI Weights", value=True)
ivrv_n_days = st.sidebar.slider("N-day period for IV/RV analysis", 7, 180, 30, 1)
run_btn = st.sidebar.button("Run Analysis", use_container_width=True, type="primary", disabled=not sel_expiry)

# --- Main Application ---
end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
with st.spinner("Fetching Kraken BTC/USD data..."):
    df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values
    times_pd = df['datetime']
    times = (times_pd - times_pd.iloc[0]).dt.total_seconds() / (24 * 3600)
    T = times.iloc[-1] if not times.empty else 1.0
    N = len(prices)
    p0 = prices[0] if len(prices) > 0 else 70000
    returns = 100 * df['close'].pct_change().dropna()
    current_price = df['close'].iloc[-1] if not df['close'].empty else 70000

    # Estimate initial volatility using GARCH or empirical
    if returns.empty:
        st.error("No valid returns data. Using default volatility.")
        V0 = 0.04
    else:
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
            V0 = (model.conditional_volatility[-1] / 100) ** 2
        except:
            V0 = (returns.std() / 100) ** 2

    if sel_expiry and run_btn:
        st.header("Stochastic Dynamics and Options Analysis")
        with st.spinner("Fetching options data..."):
            df_options = get_thalex_options_data(coin, sel_expiry, all_instruments)
        if df_options is not None and not df_options.empty:
            expiry_dt = datetime.strptime(sel_expiry, "%d%b%y").replace(
                hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
            ttm = max((expiry_dt - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600), 1e-9)
            perp_ticker = get_thalex_ticker(f"{coin}-PERPETUAL")
            spot_price = float(perp_ticker['mark_price']) if perp_ticker and perp_ticker.get('mark_price') else current_price
            forward_price = spot_price * np.exp(r_rate * ttm)
            atm_iv = df_options.iloc[(df_options['strike'] - forward_price).abs().argsort()[:1]]['iv'].iloc[0]
            if pd.isna(atm_iv) or atm_iv <= 0:
                atm_iv = np.sqrt(V0)

            # Simulate stochastic dynamics
            with st.spinner("Simulating stochastic price paths..."):
                S, V, eta = simulate_non_equilibrium(
                    S0=spot_price, V0=V0, eta0=0.0, mu=mu, phi=phi, epsilon=epsilon, lambda_=lambda_,
                    chi=chi, alpha=alpha, eta_star=eta_star, S_u=S_u, S_l=S_l, kappa=kappa,
                    rho_XY=rho_XY, rho_XZ=rho_XZ, rho_YZ=rho_YZ, T=ttm, N=252, n_paths=100
                )

            # Create DataFrame for simulated paths
            t_eval = np.linspace(0, ttm, 252 + 1)
            mean_S = np.mean(S, axis=0)
            mean_V = np.mean(V, axis=0)
            mean_eta = np.mean(eta, axis=0)
            stochastic_df = pd.DataFrame({
                "Time": t_eval,
                "Price": mean_S,
                "Volatility": np.sqrt(mean_V),
                "Arbitrage Return": mean_eta,
                "Path": "Stochastic Mean"
            })

            # Plot price and arbitrage return
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Price Path and Stochastic Dynamics")
                price_df = pd.DataFrame({
                    "Time": times,
                    "Price": prices,
                    "Path": "Historical Price"
                })
                combined_df = pd.concat([price_df, stochastic_df[['Time', 'Price', 'Path']]], ignore_index=True)
                base = alt.Chart(combined_df).encode(
                    x=alt.X("Time:Q", title="Time (days)"),
                    y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)),
                    color=alt.Color("Path:N", title="Path Type", scale=alt.Scale(domain=["Historical Price", "Stochastic Mean"], range=["blue", "orange"]))
                )
                price_line = base.mark_line(strokeWidth=2).encode(detail='Path:N')
                support_df = pd.DataFrame({"Price": [S_l]})
                resistance_df = pd.DataFrame({"Price": [S_u]})
                support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
                resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                chart = (price_line + support_lines + resistance_lines).properties(
                    title="Price Path, Stochastic Mean, and S/R Bounds", height=500
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

            with col2:
                st.subheader("Arbitrage Return Dynamics")
                eta_chart = alt.Chart(stochastic_df).mark_line(strokeWidth=2).encode(
                    x=alt.X("Time:Q", title="Time (days)"),
                    y=alt.Y("Arbitrage Return:Q", title="η"),
                    color=alt.value("purple")
                )
                eta_lines = alt.Chart(pd.DataFrame({"eta_star": [eta_star, -eta_star]})).mark_rule(stroke="red", strokeDash=[5, 5]).encode(y="eta_star:Q")
                eta_chart = (eta_chart + eta_lines).properties(
                    title="Arbitrage Return with Thresholds", height=300
                ).interactive()
                st.altair_chart(eta_chart, use_container_width=True)

            # SVI Calibration and PDF
            with st.spinner("Calibrating SVI model..."):
                market_strikes = df_options['strike'].values
                market_ivs = df_options['iv'].values
                valid_mask = pd.notna(market_ivs) & pd.notna(market_strikes) & (market_ivs > 0.001)
                if not np.any(valid_mask):
                    st.error("No valid market data for SVI calibration.")
                    svi_params, svi_error = None, np.inf
                else:
                    market_strikes = market_strikes[valid_mask]
                    market_ivs = market_ivs[valid_mask]
                    oi_weights = df_options['open_interest'].reindex(df_options.index[valid_mask]).values
                    weights = np.ones_like(market_ivs) / len(market_ivs)
                    if use_oi_weights and np.sum(oi_weights) > 0:
                        tmp_w = oi_weights / np.sum(oi_weights)
                        avg_w = 1.0 / len(tmp_w)
                        tmp_w = np.minimum(tmp_w, 5 * avg_w)
                        if np.sum(tmp_w) > 0:
                            weights = tmp_w / np.sum(tmp_w)
                    svi_params, svi_error = calibrate_raw_svi(market_ivs, market_strikes, forward_price, ttm, weights=weights)

            if svi_params:
                # Generate PDF using SVI
                price_grid = np.linspace(max(1, forward_price * 0.3), forward_price * 2.0, 300)
                log_moneyness = np.log(np.maximum(price_grid, 1e-6) / forward_price)
                svi_total_var = raw_svi_total_variance(log_moneyness, svi_params)
                svi_ivs = np.sqrt(np.maximum(svi_total_var, 1e-9) / ttm)
                call_prices_svi = np.array([BlackScholes(ttm, K, forward_price, iv, r_rate).calculate_prices()[0]
                                           for K, iv in zip(price_grid, svi_ivs)])
                pdf_df = get_pdf_from_svi_prices(price_grid, call_prices_svi, r_rate, ttm)
                u = pdf_df['pdf'].values

                # Alternative: Monte Carlo PDF from simulated paths
                final_prices = S[:, -1]
                kde = gaussian_kde(final_prices)
                mc_pdf = kde(price_grid)
                mc_pdf /= np.trapz(mc_pdf, price_grid)  # Normalize
                pdf_df['mc_pdf'] = mc_pdf

                # S/R Levels from Stochastic Dynamics
                price_std = forward_price * atm_iv * np.sqrt(ttm)
                epsilon = epsilon_factor * price_std
                peaks, _ = find_peaks(u, height=np.percentile(u, 75), distance=len(price_grid)//50)
                if len(peaks) < 4:
                    peaks, _ = find_peaks(u, height=0.01 * u.max(), distance=len(price_grid)//100)
                levels = price_grid[peaks]
                if len(peaks) < 4:
                    X = price_grid.reshape(-1, 1)
                    db = DBSCAN(eps=price_std / 2, min_samples=50).fit(X)
                    labels = db.labels_
                    unique_labels = set(labels) - {-1}
                    if unique_labels:
                        cluster_centers = [np.mean(price_grid[labels == label]) for label in unique_labels]
                        levels = np.sort(cluster_centers)[:6]
                    else:
                        top_indices = np.argsort(u)[-4:]
                        levels = np.sort(price_grid[top_indices])
                median_of_peaks = np.median(levels)
                support_levels = levels[levels <= median_of_peaks][:2]
                resistance_levels = levels[levels > median_of_peaks][-2:]

                # Probability Range
                with col2:
                    st.subheader("Options-Implied Probability Range")
                    confidence_level = st.session_state.get('profitability_threshold', 0.68)
                    lower_prob_range, upper_prob_range = calculate_probability_range(price_grid, u, confidence_level)
                    if pd.notna(lower_prob_range) and pd.notna(upper_prob_range):
                        range_cols = st.columns(2)
                        range_cols[0].metric(
                            label=f"Lower Bound ({((1-confidence_level)/2)*100:.1f}th percentile)",
                            value=f"${lower_prob_range:,.2f}"
                        )
                        range_cols[1].metric(
                            label=f"Upper Bound ({(1-((1-confidence_level)/2))*100:.1f}th percentile)",
                            value=f"${upper_prob_range:,.2f}"
                        )
                    else:
                        st.warning("Could not calculate probability range.")

                # Plot PDF with S/R
                with col2:
                    st.subheader("S/R Probabilities (SVI-Based)")
                    support_probs = get_hit_prob(support_levels, price_grid, u, None, ttm, epsilon, stochastic_df)
                    resistance_probs = get_hit_prob(resistance_levels, price_grid, u, None, ttm, epsilon, stochastic_df)
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.markdown("**Support Levels**")
                        support_data = pd.DataFrame({'Level': support_levels, 'Hit %': support_probs})
                        st.dataframe(support_data.style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))

                    with sub_col2:
                        st.markdown("**Resistance Levels**")
                        resistance_data = pd.DataFrame({'Level': resistance_levels, 'Hit %': resistance_probs})
                        st.dataframe(resistance_data.style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))

                    with st.expander("Probability Distribution with S/R Zones", expanded=True):
                        interactive_density_fig = create_interactive_density_chart(
                            price_grid, u, support_levels, resistance_levels, epsilon, forward_price,
                            prob_range=(lower_prob_range, upper_prob_range), confidence_level=confidence_level
                        )
                        st.plotly_chart(interactive_density_fig, use_container_width=True)

                # Volume Profile, IV/RV, RSI, and Options Chain
                # [Retain existing code for these sections, as they are compatible]

                # Export Results
                st.header("Export Results")
                export_button = st.button("Download Charts and Data")
                if export_button:
                    with st.spinner("Generating exports..."):
                        chart.save("price_chart.html")
                        with open("price_chart.html", "rb") as f:
                            st.download_button("Download Price Chart", f, file_name="price_chart.html")
                        if interactive_density_fig:
                            interactive_density_fig.write_html("density_chart.html")
                            with open("density_chart.html", "rb") as f:
                                st.download_button("Download Density Chart", f, file_name="density_chart.html")
                        # [Add other exports as in original code]

else:
    st.error("Could not load or process spot data. Check parameters or try again.")
