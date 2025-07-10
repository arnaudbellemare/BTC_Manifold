import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from arch import arch_model
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
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
st.set_page_config(layout="wide", page_icon="📊", page_title="BTC Options and Stochastic Dynamics Analysis")
st.title("BTC/USD Options and Non-Equilibrium Stochastic Dynamics Analysis")
st.markdown("""
This application combines options-based probability analysis with a non-equilibrium stochastic model for BTC/USD.
- **Options Analysis**: Implied volatility (IV) smiles and PDFs using SVI model.
- **Stochastic Dynamics**: Models price, volatility, and arbitrage return using SDEs with non-equilibrium effects.
- **IV/RV & Momentum**: Compares implied vs. realized volatility and includes RSI signals.
- **Volume Profile**: Historical trading activity with S/R levels and POC.
*Use the sidebar to adjust parameters or override calibrated values.*
""")

# --- Thalex API Configuration ---
THALEX_BASE_URL = "https://thalex.com/api/v2"
REQUEST_TIMEOUT = 15
Z_SCORE_HIGH_IV_THRESHOLD = 1.0
Z_SCORE_LOW_IV_THRESHOLD = -1.0
RSI_BULLISH_THRESHOLD = 65
RSI_BEARISH_THRESHOLD = 35

# --- Stochastic Dynamics Simulation ---
def simulate_non_equilibrium(S0, V0, eta0, mu, phi, epsilon, lambda_, chi, alpha, eta_star, S_u, S_l, kappa, rho_XY, rho_XZ, rho_YZ, T, N, n_paths=100):
    dt = T / N
    S = np.zeros((n_paths, N+1))
    V = np.zeros((n_paths, N+1))
    eta = np.zeros((n_paths, N+1))
    S[:, 0] = S0
    V[:, 0] = V0
    eta[:, 0] = eta0

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

        lambda_eff = np.where(np.abs(eta_t) <= eta_star, lambda_ * exp_eta, lambda_ * 0.1 * exp_eta)
        dS = mu * (1 - alpha * eta_ratio) * S_t * dt + np.sqrt(V_t) * S_t * dW_correlated[:, 0]
        dV = phi * (V0 * exp_eta - V_t) * dt + epsilon * exp_eta * np.sqrt(V_t) * dW_correlated[:, 1]
        d_eta = (-lambda_eff * eta_t + kappa * exp_bound) * dt + chi * dW_correlated[:, 2]

        S[:, t+1] = S_t + dS
        V[:, t+1] = np.maximum(V_t + dV, 1e-6)
        eta[:, t+1] = eta_t + d_eta

    logging.info(f"Simulation stats - Mean eta: {np.mean(eta):.4f}, Max |eta|: {np.max(np.abs(eta)):.4f}, Non-eq count: {np.sum(np.abs(eta) > eta_star)}")
    return S, V, eta

# --- Helper Functions (omitted for brevity, use previous versions) ---
# fetch_kraken_data, get_thalex_instruments, simulate_instruments, get_thalex_options_data,
# get_thalex_ticker, simulate_options_data, get_expiries_from_instruments, BlackScholes,
# raw_svi_total_variance, check_raw_svi_params_validity, check_svi_butterfly_arbitrage_gk,
# svi_objective_function, calibrate_raw_svi, get_pdf_from_svi_prices, calculate_probability_range,
# create_interactive_density_chart, create_volume_profile_chart

# --- Sidebar Configuration ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 90, 30)
epsilon_factor = st.sidebar.slider("S/R Zone Width Factor", 0.1, 2.0, 0.5, step=0.05)
st.session_state['profitability_threshold'] = st.sidebar.slider("Profitability Confidence Interval (%)", 68, 99, 68, step=1) / 100.0

# Calibrate stochastic parameters
end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
with st.spinner("Fetching Kraken BTC/USD data..."):
    df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values
    times_pd = df['datetime']
    times = (times_pd - times_pd.iloc[0]).dt.total_seconds() / (24 * 3600)
    T = times.iloc[-1] if not times.empty else 1.0
    returns = 100 * df['close'].pct_change().dropna()
    current_price = df['close'].iloc[-1] if not df['close'].empty else 65000
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    mu = log_returns.mean() * 365 if not log_returns.empty else 0.05
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off', show_warning=False)
        V0 = (model.conditional_volatility[-1] / 100) ** 2 if model.convergence_flag == 0 else 0.04
        phi = 1 - (model.params['alpha[1]'] + model.params['beta[1]']) if model.convergence_flag == 0 else 1.0
        sigma = model.conditional_volatility / 100
        epsilon = sigma.std() * np.sqrt(365) if len(sigma) > 1 else 0.2
        logging.info(f"GARCH fit successful - V0: {V0:.4f}, phi: {phi:.4f}, epsilon: {epsilon:.4f}")
    except Exception as e:
        V0 = (returns.std() / 100) ** 2 if not returns.empty else 0.04
        phi = 1.0
        epsilon = 0.2
        sigma = np.full_like(returns, returns.std() / 100 if not returns.empty else 0.02)
        logging.warning(f"GARCH fit failed: {e}. Using fallback - V0: {V0:.4f}, phi: {phi:.4f}, epsilon: {epsilon:.4f}")
    lambda_ = np.log(2) / 1.0
    chi = 0.1 * epsilon
    alpha = 0.5
    eta_star = 0.09
    kappa = 0.1
    if len(sigma) > len(log_returns):
        sigma = sigma[-len(log_returns):]
    elif len(sigma) < len(log_returns):
        sigma = np.pad(sigma, (0, len(log_returns) - len(sigma)), mode='edge')
    rho_XY = np.corrcoef(log_returns, sigma)[0, 1] if len(log_returns) > 1 and len(sigma) > 1 and not np.isnan(sigma).all() else 0.3
    logging.info(f"Correlation data - log_returns len: {len(log_returns)}, sigma len: {len(sigma)}, rho_XY: {rho_XY:.4f}")
    rho_XZ = 0.2
    rho_YZ = 0.0

st.sidebar.header("Stochastic Dynamics Parameters (Override)")
override_params = st.sidebar.checkbox("Manually Override Parameters", value=False)
if override_params:
    mu = st.sidebar.slider("Price Drift (μ)", 0.0, 0.2, mu, step=0.01)
    phi = st.sidebar.slider("Volatility Mean Reversion (φ)", 0.1, 2.0, phi, step=0.1)
    epsilon = st.sidebar.slider("Volatility of Volatility (ε)", 0.05, 0.5, epsilon, step=0.01)
    lambda_ = st.sidebar.slider("Arbitrage Mean Reversion (λ)", 0.1, 2.0, lambda_, step=0.1)
    chi = st.sidebar.slider("Arbitrage Volatility (χ)", 0.01, 0.1, chi, step=0.01)
    alpha = st.sidebar.slider("Mispricing Impact (α)", 0.1, 1.0, alpha, step=0.1)
    eta_star = st.sidebar.slider("Mispricing Threshold (η*)", 0.01, 0.2, eta_star, step=0.01)
    kappa = st.sidebar.slider("Arbitrage Revival (κ)", 0.01, 0.5, kappa, step=0.01)
    rho_XY = st.sidebar.slider("Price-Volatility Correlation (ρ_XY)", -0.5, 0.5, rho_XY, step=0.05)
    rho_XZ = st.sidebar.slider("Price-Arbitrage Correlation (ρ_XZ)", -0.5, 0.5, rho_XZ, step=0.05)
    rho_YZ = st.sidebar.slider("Volatility-Arbitrage Correlation (ρ_YZ)", -0.5, 0.5, rho_YZ, step=0.05)

st.sidebar.header("Options Analysis Parameters")
all_instruments = get_thalex_instruments()
coin = "BTC"
expiries = get_expiries_from_instruments(all_instruments, coin)
sel_expiry = st.sidebar.selectbox("Options Expiry", expiries, index=0) if expiries else None
r_rate = st.sidebar.slider("Prime Rate (%)", 0.0, 14.0, 1.6, 0.1) / 100.0
use_oi_weights = st.sidebar.checkbox("Use OI Weights", value=True)
run_btn = st.sidebar.button("Run Analysis", use_container_width=True, type="primary", disabled=not sel_expiry)

# --- Main Application ---
if df is not None and len(df) > 10 and sel_expiry and run_btn:
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
        S_u = forward_price + 2 * forward_price * atm_iv * np.sqrt(ttm)
        S_l = max(forward_price - 2 * forward_price * atm_iv * np.sqrt(ttm), 1e-6)

        with st.spinner("Simulating stochastic price paths..."):
            S, V, eta = simulate_non_equilibrium(
                S0=spot_price, V0=V0, eta0=0.05, mu=mu, phi=phi, epsilon=epsilon, lambda_=lambda_,
                chi=chi, alpha=alpha, eta_star=eta_star, S_u=S_u, S_l=S_l, kappa=kappa,
                rho_XY=rho_XY, rho_XZ=rho_XZ, rho_YZ=rho_YZ, T=ttm, N=252
            )

        t_eval = np.linspace(0, ttm, 252 + 1)
        stochastic_df = pd.DataFrame({
            "Time": t_eval,
            "Price": np.mean(S, axis=0),
            "Volatility": np.sqrt(np.mean(V, axis=0)),
            "Arbitrage Return": np.mean(eta, axis=0),
            "Path": "Stochastic Mean"
        })

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
            price_grid = np.linspace(max(1, forward_price * 0.3), forward_price * 2.0, 300)
            log_moneyness = np.log(np.maximum(price_grid, 1e-6) / forward_price)
            svi_total_var = raw_svi_total_variance(log_moneyness, svi_params)
            svi_ivs = np.sqrt(np.maximum(svi_total_var, 1e-9) / ttm)
            call_prices_svi = np.array([BlackScholes(ttm, K, forward_price, iv, r_rate).calculate_prices()[0]
                                       for K, iv in zip(price_grid, svi_ivs)])
            pdf_df = get_pdf_from_svi_prices(price_grid, call_prices_svi, r_rate, ttm)
            u = pdf_df['pdf'].values

            final_prices = S[:, -1]
            kde = gaussian_kde(final_prices)
            mc_pdf = kde(price_grid)
            mc_pdf /= np.trapz(mc_pdf, price_grid)
            pdf_df['mc_pdf'] = mc_pdf

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

            confidence_level = st.session_state.get('profitability_threshold', 0.68)
            lower_prob_range, upper_prob_range = calculate_probability_range(price_grid, u, confidence_level)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Price Path and Stochastic Dynamics")
                if len(times) != len(prices):
                    st.warning(f"Data mismatch: Truncating times to match prices length ({len(prices)})")
                    times = times[:len(prices)]
                if len(t_eval) != len(stochastic_df['Price']):
                    st.error(f"Simulation data mismatch: t_eval length ({len(t_eval)}) != stochastic_df Price length ({len(stochastic_df['Price'])})")
                    t_eval = t_eval[:len(stochastic_df['Price'])]
                
                price_df = pd.DataFrame({"Time": times, "Price": prices, "Path": "Historical Price"})
                stochastic_df['Time'] = t_eval
                combined_df = pd.concat([price_df, stochastic_df[['Time', 'Price', 'Path']]], ignore_index=True)
                
                base = alt.Chart(combined_df).encode(
                    x=alt.X("Time:Q", title="Time (days)", scale=alt.Scale(domain=[0, max(times.max(), ttm)+1])),
                    y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False, domain=[S_l-5000, S_u+5000])),
                    color=alt.Color("Path:N", scale=alt.Scale(domain=["Historical Price", "Stochastic Mean"], range=["blue", "orange"]))
                )
                price_line = base.mark_line(strokeWidth=2).encode(detail='Path:N')
                
                support_df = pd.DataFrame({"Price": [S_l]})
                resistance_df = pd.DataFrame({"Price": [S_u]})
                support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
                resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                
                if pd.notna(lower_prob_range) and pd.notna(upper_prob_range):
                    prob_range_df = pd.DataFrame({
                        'lower_bound': [lower_prob_range],
                        'upper_bound': [upper_prob_range],
                        'label': [f"{confidence_level:.0%} Range"]
                    })
                    prob_band = alt.Chart(prob_range_df).mark_rect(opacity=0.1, color='yellow').encode(
                        y='lower_bound:Q',
                        y2='upper_bound:Q'
                    ).properties(layer="back")
                
                chart = (prob_band + price_line + support_lines + resistance_lines).properties(
                    title="Price Path, Stochastic Mean, and S/R Bounds with Prob Range",
                    height=500
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
                eta_chart = (eta_chart + eta_lines).properties(title="Arbitrage Return with Thresholds", height=300).interactive()
                st.altair_chart(eta_chart, use_container_width=True)

                st.subheader("Options-Implied Probability Range")
                if pd.notna(lower_prob_range) and pd.notna(upper_prob_range):
                    range_cols = st.columns(2)
                    range_cols[0].metric(label=f"Lower Bound ({((1-confidence_level)/2)*100:.1f}th percentile)", value=f"${lower_prob_range:,.2f}")
                    range_cols[1].metric(label=f"Upper Bound ({(1-((1-confidence_level)/2))*100:.1f}th percentile)", value=f"${upper_prob_range:,.2f}")
                else:
                    st.warning("Could not calculate probability range.")

                st.subheader("S/R Probabilities (SVI-Based)")
                support_probs = [np.trapz(u[(price_grid >= l - epsilon) & (price_grid <= l + epsilon)], price_grid[(price_grid >= l - epsilon) & (price_grid <= l + epsilon)]) for l in support_levels]
                resistance_probs = [np.trapz(u[(price_grid >= l - epsilon) & (price_grid <= l + epsilon)], price_grid[(price_grid >= l - epsilon) & (price_grid <= l + epsilon)]) for l in resistance_levels]
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

            st.header("Volume Profile and Options Chain")
            volume_profile_fig, poc = create_volume_profile_chart(df, support_levels, resistance_levels, epsilon, current_price)
            if volume_profile_fig:
                st.plotly_chart(volume_profile_fig, use_container_width=True)
                st.metric("Point of Control (POC)", f"${poc['price_bin']:,.2f}")
                st.metric("Current Price", f"${current_price:,.2f}")

            # [Add options chain, IV/RV, RSI, and export sections as needed]
else:
    st.error("Could not load or process spot data. Check parameters or try again.")
