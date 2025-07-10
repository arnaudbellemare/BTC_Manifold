where is this ? i just wanted to have the simulated path import streamlit as st
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
This application models BTC/USD options under non-equilibrium conditions, capturing market instability (e.g., crashes) with a stochastic dynamics model.
- **Options Pricing**: Prices options using SVI and non-equilibrium dynamics, accounting for arbitrage opportunities.
- **Probability Forecasts**: Estimates price ranges and support/resistance levels for risk management.
- **Trading Signals**: Highlights mispricing (η) and volatility mismatches for arbitrage or directional trades.
- **Volume Profile**: Identifies key price levels (POC, S/R) from historical trading activity.
*Adjust sidebar parameters to optimize model dynamics and outputs.*
""")

# --- Thalex API Configuration ---
THALEX_BASE_URL = "https://thalex.com/api/v2"
REQUEST_TIMEOUT = 15
Z_SCORE_HIGH_IV_THRESHOLD = 1.0
Z_SCORE_LOW_IV_THRESHOLD = -1.0
RSI_BULLISH_THRESHOLD = 65
RSI_BEARISH_THRESHOLD = 35

# --- Stochastic Dynamics Simulation ---
def simulate_non_equilibrium(S0, V0, eta0, mu, phi, epsilon, lambda_, chi, alpha, eta_star, S_u, S_l, kappa, rho_XY, rho_XZ, rho_YZ, T, N, n_paths=2000):
    dt = T / N
    S = np.zeros((n_paths, N+1))
    V = np.zeros((n_paths, N+1))
    eta = np.zeros((n_paths, N+1))
    S[:, 0] = max(S0, 1e-6)
    V[:, 0] = max(V0, 1e-6)
    eta[:, 0] = np.clip(eta0, -0.5, 0.5)

    # Validate correlation coefficients
    rho_XY = np.clip(rho_XY, -0.99, 0.99) if np.isfinite(rho_XY) else 0.3
    rho_XZ = np.clip(rho_XZ, -0.99, 0.99) if np.isfinite(rho_XZ) else 0.2
    rho_YZ = np.clip(rho_YZ, -0.99, 0.99) if np.isfinite(rho_YZ) else 0.0

    # Construct and validate correlation matrix
    corr_matrix = np.array([[1.0, rho_XY, rho_XZ],
                            [rho_XY, 1.0, rho_YZ],
                            [rho_XZ, rho_YZ, 1.0]])
    if not np.all(np.isfinite(corr_matrix)) or not np.all(np.linalg.eigvals(corr_matrix) > 0):
        logging.warning("Correlation matrix invalid or not positive definite. Using identity matrix.")
        corr_matrix = np.eye(3)
    L = np.linalg.cholesky(corr_matrix)

    for t in range(N):
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, 3))
        dW_correlated = dW @ L.T

        S_t = S[:, t]
        V_t = V[:, t]
        eta_t = eta[:, t]
        eta_ratio = np.clip(eta_t / eta_star, -2, 2)
        exp_eta = np.exp(np.clip(-eta_ratio**2, -700, 0))

        # Price bound term
        price_bound_term = (2 * S_t / S_u - S_l / S_u - 1)**2
        exp_bound = 1 - np.exp(np.clip(-price_bound_term, -700, 0))

        # Mean reversion rate
        lambda_eff = lambda_ * exp_eta

        # SDEs
        mu_t = np.clip(mu * (1 - alpha * eta_ratio), -0.3, 0.3)
        dS = mu_t * S_t * dt + np.sqrt(np.maximum(V_t, 1e-6)) * S_t * dW_correlated[:, 0]
        dV = phi * (np.maximum(V0 * exp_eta, 1e-6) - V_t) * dt + epsilon * exp_eta * np.sqrt(np.maximum(V_t, 1e-6)) * dW_correlated[:, 1]
        d_eta = (-lambda_eff * eta_t + kappa * exp_bound) * dt + chi * dW_correlated[:, 2]

        S[:, t+1] = np.clip(S_t + dS, 1e-6, 1e7)
        V[:, t+1] = np.maximum(V_t + dV, 1e-6)
        eta[:, t+1] = np.clip(eta_t + d_eta, -1.0, 1.0)

        if t % 100 == 0:
            logging.debug(f"Step {t}: S_mean={np.mean(S_t):.2f}, V_mean={np.mean(V_t):.4f}, eta_mean={np.mean(eta_t):.4f}")

    final_prices = S[:, -1]
    valid_final_prices = final_prices[np.isfinite(final_prices)]
    logging.info(f"Simulation stats - Mean eta: {np.mean(eta):.4f}, Max |eta|: {np.max(np.abs(eta)):.4f}, "
                 f"Valid final prices: {len(valid_final_prices)}, Mean S_final: {np.mean(valid_final_prices):.2f}")
    return S, V, eta

# --- Helper Functions ---
@st.cache_data
def fetch_kraken_data(symbol, timeframe, start_date, end_date):
    exchange = ccxt.kraken()
    since = int(start_date.timestamp() * 1000)
    timeframe_seconds = 3600
    all_ohlcv = []
    max_retries = 5
    progress_bar = st.progress(0)
    while since < int(end_date.timestamp() * 1000):
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=500)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = int(ohlcv[-1][0]) + timeframe_seconds * 1000
                progress_bar.progress(min(1.0, since / (end_date.timestamp() * 1000)))
                logging.info(f"Kraken fetch attempt {attempt+1} succeeded at timestamp {since}")
                time.sleep(3)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed to fetch Kraken data after {max_retries} attempts: {e}")
                    logging.error(f"Kraken fetch failed: {e}")
                    break
                time.sleep(2 ** (attempt + 2))
    progress_bar.empty()
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
        if len(df) >= 10:
            logging.info(f"Kraken data fetched: {len(df)} records")
            return df
    st.error("Failed to fetch Kraken data. Using simulated data.")
    sim_t = pd.date_range(start=start_date, end=end_date, freq='h')
    n = len(sim_t)
    vol = np.random.normal(0, 0.03, n)
    vol = 0.02 + 0.01 * np.exp(-np.arange(n)/100) * np.cumsum(vol)
    sim_prices = 111000 * np.exp(np.cumsum(vol * np.random.normal(0, 1, n)))
    logging.warning("Using simulated Kraken data with initial price 111000")
    return pd.DataFrame({'datetime': sim_t, 'close': sim_prices, 'volume': np.random.randint(50, 200, n)})

@st.cache_data(ttl=300)
def get_thalex_instruments():
    url = f"{THALEX_BASE_URL}/public/instruments"
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
    def fetch():
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Thalex API response: {data.get('result', 'No result key')[:100]}...")
        return data
    try:
        data = fetch()
        instruments = data.get("result", [])
        if not instruments:
            st.warning("Thalex API returned empty instruments list. Using simulated data.")
            return simulate_instruments()
        return instruments
    except Exception as e:
        st.error(f"Failed to fetch Thalex instruments after retries: {e}")
        logging.error(f"Thalex instruments fetch error: {e}")
        return simulate_instruments()

def simulate_instruments():
    return [
        {"instrument_name": f"BTC-{d}-111000-C", "type": "option"} for d in ["31DEC25", "31MAR26"]
    ]

@st.cache_data(ttl=300)
def get_thalex_options_data(coin: str, expiry_str: str, instruments: List[Dict]):
    instrument_names = [
        instr['instrument_name'] for instr in instruments
        if instr.get('type') == 'option' and instr.get('instrument_name', '').startswith(f"{coin.upper()}-{expiry_str}-")
    ]
    if not instrument_names:
        st.warning(f"No options found for {coin}-{expiry_str}. Using simulated data.")
        return simulate_options_data(coin)
    options_data = []
    progress_bar = st.progress(0)
    for i, name in enumerate(instrument_names):
        ticker = get_thalex_ticker(name)
        if ticker and ticker.get('mark_price') is not None and ticker.get('iv') is not None:
            try:
                parts = name.split('-')
                options_data.append({
                    'instrument': name,
                    'strike': float(parts[2]),
                    'type': parts[3],
                    'mark_price': float(ticker['mark_price']),
                    'iv': float(ticker['iv']),
                    'open_interest': float(ticker.get('open_interest', 0)),
                    'delta': float(ticker.get('delta', np.nan))
                })
            except (ValueError, IndexError):
                continue
        time.sleep(0.05)
        progress_bar.progress((i + 1) / len(instrument_names))
    progress_bar.empty()
    if not options_data:
        st.warning(f"No valid options data for {coin}-{expiry_str}. Using simulated data.")
        return simulate_options_data(coin)
    df = pd.DataFrame(options_data)
    df = df[(df['iv'] > 0.0001) & (df['mark_price'] > 0)].copy()
    return df

def get_thalex_ticker(instrument_name: str):
    url = f"{THALEX_BASE_URL}/public/ticker"
    params = {"instrument_name": instrument_name}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("result", {})
    except Exception:
        return None

def simulate_options_data(coin: str) -> pd.DataFrame:
    strikes = np.linspace(90000, 130000, 20)
    types = ['C', 'P']
    sim_data = []
    for s in strikes:
        for t in types:
            iv = 0.7 + np.random.normal(0, 0.15)
            price = s * iv * 0.05
            sim_data.append({
                'instrument': f"{coin}-{s}-{t}",
                'strike': s,
                'type': t,
                'mark_price': price,
                'iv': iv,
                'open_interest': np.random.randint(50, 500),
                'delta': np.random.uniform(-1, 1)
            })
    return pd.DataFrame(sim_data)

def get_expiries_from_instruments(instruments: List[Dict], coin: str) -> List[str]:
    expiry_dates = set()
    now_utc = datetime.now(timezone.utc)
    date_pattern = re.compile(rf"^{re.escape(coin.upper())}-(\d{{2}}[A-Z]{{3}}\d{{2}})-.*")
    for instr in instruments:
        if instr.get('type') != 'option':
            continue
        name = instr.get('instrument_name', '')
        match = date_pattern.match(name)
        if match:
            expiry_str = match.group(1)
            try:
                expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(
                    hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                )
                if expiry_dt > now_utc:
                    expiry_dates.add(expiry_str)
            except ValueError:
                continue
    return sorted(list(expiry_dates), key=lambda d: datetime.strptime(d, "%d%b%y"))

class BlackScholes:
    def __init__(self, time_to_maturity, strike, underlying_price, volatility, interest_rate):
        self.T = max(float(time_to_maturity), 1e-9)
        self.K = max(float(strike), 1e-6)
        self.S = max(float(underlying_price), 1e-6)
        self.sigma = max(float(volatility), 1e-6)
        self.r = float(interest_rate)
        self.sigma_sqrt_T = self.sigma * np.sqrt(self.T)
        if self.sigma_sqrt_T < 1e-9:
            if self.S > self.K:
                self.d1, self.d2 = np.inf, np.inf
            elif self.S < self.K:
                self.d1, self.d2 = -np.inf, -np.inf
            else:
                self.d1, self.d2 = 0, 0
        else:
            self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / self.sigma_sqrt_T
            self.d2 = self.d1 - self.sigma_sqrt_T

    def calculate_prices(self):
        if self.sigma_sqrt_T < 1e-9:
            call_price = max(0, self.S - self.K * np.exp(-self.r * self.T))
            put_price = max(0, self.K * np.exp(-self.r * self.T) - self.S)
        else:
            call_price = self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
        return max(0, call_price), max(0, put_price)

def raw_svi_total_variance(k, params):
    a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']
    b = max(0, b)
    rho = np.clip(rho, -0.9999, 0.9999)
    sigma = max(1e-5, sigma)
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def check_raw_svi_params_validity(params):
    a, b, rho, sigma = params['a'], params.get('b', 0), params.get('rho', 0), params.get('sigma', 1e-3)
    if b < 0 or not (-1 < rho < 1) or sigma <= 0:
        return False
    return (a + b * sigma * np.sqrt(1 - rho**2)) >= -1e-9

def check_svi_butterfly_arbitrage_gk(k_val, svi_params, T_expiry):
    w_func = lambda x_k_internal: raw_svi_total_variance(x_k_internal, svi_params)
    try:
        w_k = w_func(k_val)
        if w_k <= 1e-9:
            return -1e9
        eps = 1e-5
        w_k_plus_eps = w_func(k_val + eps)
        w_k_minus_eps = w_func(k_val - eps)
        dw_dk = (w_k_plus_eps - w_k_minus_eps) / (2 * eps)
        d2w_dk2 = (w_k_plus_eps - 2 * w_k + w_k_minus_eps) / (eps**2)
    except:
        logging.error(f"Error in numerical differentiation for g(k) (k={k_val})")
        return -1e9
    term1_denom = 2 * w_k
    if abs(term1_denom) < 1e-9:
        return -1e9
    term1_num_factor = k_val * dw_dk / term1_denom
    term1 = (1 - term1_num_factor)**2
    term2_factor = (dw_dk / 2)**2
    term2_bracket_denom = w_k
    if abs(term2_bracket_denom) < 1e-9:
        return -1e9
    term2_bracket = (1 / term2_bracket_denom) + 0.25
    term2 = term2_factor * term2_bracket
    term3 = d2w_dk2 / 2.0
    g_k = term1 - term2 + term3
    return g_k

def svi_objective_function(svi_params_array, market_ivs, market_ks, T, F, weights):
    params = {'a': svi_params_array[0], 'b': svi_params_array[1], 'rho': svi_params_array[2],
              'm': svi_params_array[3], 'sigma': svi_params_array[4]}
    if not check_raw_svi_params_validity(params):
        return 1e12
    model_total_variances = raw_svi_total_variance(market_ks, params)
    if np.any(model_total_variances < 0):
        return 1e10 + np.sum(np.abs(model_total_variances[model_total_variances < 0]))
    model_ivs = np.sqrt(np.maximum(model_total_variances, 1e-9) / T)
    error = np.sum(weights * ((model_ivs - market_ivs)**2))
    g_atm = check_svi_butterfly_arbitrage_gk(0.0, params, T)
    if g_atm < -1e-5:
        error += 1e6 * abs(g_atm)
    if params['b'] * (1 + abs(params['rho'])) * T > 4.0 + 0.1:
        error += 1e5
    return error

@st.cache_data(ttl=120)
def calibrate_raw_svi(market_ivs, market_strikes, F, T, initial_params_dict=None, weights=None):
    market_ks = np.log(np.maximum(market_strikes, 1e-6) / F)
    if weights is None:
        weights = np.ones_like(market_ivs) / len(market_ivs)
    weights = np.array(weights) / np.sum(weights)
    if initial_params_dict:
        p0 = [initial_params_dict['a'], initial_params_dict['b'], initial_params_dict['rho'],
              initial_params_dict['m'], initial_params_dict['sigma']]
    else:
        atm_total_var_approx = np.interp(0, market_ks, market_ivs**2 * T, left=(market_ivs[0]**2*T), right=(market_ivs[-1]**2*T))
        p0 = [atm_total_var_approx * 0.9, 0.15, -0.4, 0.0, 0.2]
    bounds = [(None, None), (1e-5, 2.0/T if T > 0 else 20.0), (-0.999, 0.999), (-2.0, 2.0), (1e-4, 3.0)]
    result = minimize(svi_objective_function, p0, args=(market_ivs, market_ks, T, F, weights),
                     method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 500})
    if result.success or result.status in [0, 1, 2]:
        cal_p = {'a': result.x[0], 'b': result.x[1], 'rho': result.x[2], 'm': result.x[3], 'sigma': result.x[4]}
        if not check_raw_svi_params_validity(cal_p):
            logging.warning(f"SVI Calib params invalid post-opt: {cal_p}")
        logging.info(f"SVI Calib Success: {result.message}, Error: {result.fun:.2e}, Params: {cal_p}")
        return cal_p, result.fun
    logging.error(f"SVI Calib failed: {result.message}")
    return None, np.inf

def get_pdf_from_svi_prices(K_grid_dense, call_prices_svi, r_rate, T_expiry):
    if len(K_grid_dense) != len(call_prices_svi) or len(K_grid_dense) < 3:
        return pd.DataFrame({'strike': K_grid_dense, 'pdf': np.nan})
    dK = K_grid_dense[1] - K_grid_dense[0]
    if dK <= 1e-6:
        return pd.DataFrame({'strike': K_grid_dense, 'pdf': np.nan})
    d2C_dK2 = np.zeros_like(call_prices_svi)
    d2C_dK2[1:-1] = (call_prices_svi[2:] - 2*call_prices_svi[1:-1] + call_prices_svi[:-2]) / (dK**2)
    d2C_dK2[0] = d2C_dK2[1]
    d2C_dK2[-1] = d2C_dK2[-2]
    pdf_values = np.exp(r_rate * T_expiry) * d2C_dK2
    pdf_values = np.maximum(pdf_values, 0)
    integral_pdf = np.trapz(pdf_values, K_grid_dense)
    if integral_pdf > 1e-6:
        pdf_values /= integral_pdf
    return pd.DataFrame({'strike': K_grid_dense, 'pdf': pdf_values})

def calculate_probability_range(price_grid, pdf_values, confidence_level):
    if len(price_grid) != len(pdf_values) or len(price_grid) < 2:
        return np.nan, np.nan
    integral = np.trapz(pdf_values, price_grid)
    if integral < 1e-6:
        return np.nan, np.nan
    normalized_pdf = pdf_values / integral
    cdf_values = [np.trapz(normalized_pdf[:i+1], price_grid[:i+1]) for i in range(len(price_grid))]
    cdf_values = np.array(cdf_values)
    cdf_values[0], cdf_values[-1] = 0.0, 1.0
    unique_cdf, unique_indices = np.unique(cdf_values, return_index=True)
    unique_prices = price_grid[unique_indices]
    if len(unique_cdf) < 2:
        return np.nan, np.nan
    inverse_cdf = interp1d(unique_cdf, unique_prices, bounds_error=False, fill_value="extrapolate")
    tail_prob = (1.0 - confidence_level) / 2.0
    lower_quantile, upper_quantile = tail_prob, 1.0 - tail_prob
    lower_bound = float(inverse_cdf(lower_quantile))
    upper_bound = float(inverse_cdf(upper_quantile))
    return lower_bound, upper_bound

def create_interactive_density_chart(price_grid, density, s_levels, r_levels, epsilon, forward_price, prob_range=None, confidence_level=None):
    if len(price_grid) == 0 or len(density) == 0:
        st.error("Density chart data is empty.")
        return None
    fig = go.Figure()
    if prob_range and confidence_level is not None and pd.notna(prob_range[0]) and pd.notna(prob_range[1]):
        lower, upper = prob_range
        fig.add_vrect(x0=lower, x1=upper, fillcolor="rgba(255, 255, 0, 0.2)", layer="below", line_width=0,
                      annotation_text=f"{confidence_level:.0%} Prob. Range", annotation_position="top left")
    fig.add_trace(go.Scatter(x=price_grid, y=density, mode='lines', name='Probability Density',
                            fill='tozeroy', line_color='lightblue', hovertemplate='Price: $%{x:,.2f}<br>Density: %{y:.4f}'))
    for level in s_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="green", opacity=0.2, layer="below", line_width=0)
        fig.add_vline(x=level, line_color='green', line_dash='dash')
    for level in r_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="red", opacity=0.2, layer="below", line_width=0)
        fig.add_vline(x=level, line_color='red', line_dash='dash')
    fig.add_vline(x=forward_price, line_dash="dot", line_color="lightslategrey", annotation_text="Fwd (F)")
    fig.update_layout(
        title="SVI Implied Probability Distribution with S/R Zones",
        xaxis_title="Price (USD)",
        yaxis_title="Density",
        showlegend=False,
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    return fig

def create_volume_profile_chart(df, s_levels, r_levels, epsilon, current_price, n_bins=100):
    if df.empty or 'close' not in df or 'volume' not in df:
        st.error("Volume profile data is invalid or empty.")
        return None, None
    price_range = df['close'].max() - df['close'].min()
    if price_range == 0:
        st.error("Price range is zero.")
        return None, None
    bin_size = price_range / n_bins
    df['price_bin'] = (df['close'] // bin_size) * bin_size
    volume_by_price = df.groupby('price_bin')['volume'].sum().reset_index()
    if volume_by_price.empty:
        st.error("Volume profile data is empty after grouping.")
        return None, None
    poc = volume_by_price.loc[volume_by_price['volume'].idxmax()]
    fig = go.Figure(go.Bar(
        y=volume_by_price['price_bin'],
        x=volume_by_price['volume'],
        orientation='h',
        marker_color='lightblue',
        hovertemplate='Price: $%{y:,.2f}<br>Volume: %{x:,.0f}'
    ))
    fig.add_shape(type="line", y0=poc['price_bin'], y1=poc['price_bin'], x0=0, x1=poc['volume'],
                  line=dict(color="orange", width=2, dash="dash"))
    if current_price and not np.isnan(current_price):
        fig.add_hline(y=current_price, line_color='blue', line_width=2, line_dash='solid',
                      annotation_text="Current Price", annotation_position="top right")
    for level in s_levels:
        fig.add_hrect(y0=level - epsilon, y1=level + epsilon, fillcolor="green", opacity=0.2,
                      layer="below", line_width=0, annotation_text="Support", annotation_position="top left")
        fig.add_hline(y=level, line_color='green', line_dash='dash')
    for level in r_levels:
        fig.add_hrect(y0=level - epsilon, y1=level + epsilon, fillcolor="red", opacity=0.2,
                      layer="below", line_width=0, annotation_text="Resistance", annotation_position="top left")
        fig.add_hline(y=level, line_color='red', line_dash='dash')
    fig.update_layout(
        title="Historical Volume Profile with S/R Zones and Current Price",
        xaxis_title="Volume Traded",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=400
    )
    return fig, poc

# --- Sidebar Configuration ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 180, 90)
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
    current_price = df['close'].iloc[-1] if not df['close'].empty else 111000
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    log_returns = log_returns[np.isfinite(log_returns)]  # Remove NaNs/infinities
    if len(log_returns) < 10:
        logging.warning("Insufficient valid log returns. Using simulated data.")
        log_returns = np.random.normal(0, 0.01, len(prices)-1)
    mu = log_returns.mean() * 365 if len(log_returns) > 0 else 0.2
    try:
        model = arch_model(returns[np.isfinite(returns)], vol='Garch', p=1, q=1).fit(disp='off', show_warning=False)
        if model.convergence_flag == 0:
            V0 = (model.conditional_volatility[-1] / 100) ** 2
            phi = 1 - (model.params['alpha[1]'] + model.params['beta[1]'])
            sigma = model.conditional_volatility / 100
            epsilon = sigma.std() * np.sqrt(365) if len(sigma) > 1 else 0.1
            logging.info(f"GARCH fit successful - V0: {V0:.4f}, phi: {phi:.4f}, epsilon: {epsilon:.4f}")
        else:
            raise Exception("GARCH convergence failed")
    except Exception as e:
        V0 = 0.04 if len(returns) == 0 or not np.isfinite(returns.std()) else (returns.std() / 100) ** 2
        phi = 0.5
        epsilon = 0.1
        sigma = np.full_like(returns, 0.02 if len(returns) == 0 or not np.isfinite(returns.std()) else returns.std() / 100)
        logging.warning(f"GARCH fit failed: {e}. Using fallback - V0: {V0:.4f}, phi: {phi:.4f}, epsilon: {epsilon:.4f}")
    lambda_ = np.log(2) / 1.0
    chi = 0.01
    alpha = 0.6
    eta_star = 0.09
    kappa = 0.05
    if len(sigma) > len(log_returns):
        sigma = sigma[:len(log_returns)]
    elif len(sigma) < len(log_returns):
        sigma = np.pad(sigma, (0, len(log_returns) - len(sigma)), mode='edge')
    valid_mask = np.isfinite(log_returns) & np.isfinite(sigma)
    log_returns = log_returns[valid_mask]
    sigma = sigma[valid_mask]
    if len(log_returns) > 1 and len(sigma) > 1:
        try:
            corr = np.corrcoef(log_returns, sigma)[0, 1]
            rho_XY = np.clip(corr, -0.99, 0.99) if np.isfinite(corr) else 0.3
        except:
            rho_XY = 0.3
            logging.warning("Correlation computation failed. Using rho_XY=0.3")
    else:
        rho_XY = 0.3
        logging.warning("Insufficient valid data for correlation. Using rho_XY=0.3")
    rho_XZ = 0.2
    rho_YZ = 0.0
    logging.info(f"Correlation data - log_returns len: {len(log_returns)}, sigma len: {len(sigma)}, rho_XY: {rho_XY:.4f}")

st.sidebar.header("Stochastic Dynamics Parameters (Override)")
override_params = st.sidebar.checkbox("Manually Override Parameters", value=False)
if override_params:
    mu = st.sidebar.slider("Price Drift (μ)", 0.0, 0.3, mu, step=0.01)
    phi = st.sidebar.slider("Volatility Mean Reversion (φ)", 0.1, 2.0, phi, step=0.1)
    epsilon = st.sidebar.slider("Volatility of Volatility (ε)", 0.01, 0.2, epsilon, step=0.01)
    lambda_ = st.sidebar.slider("Arbitrage Mean Reversion (λ)", 0.1, 2.0, lambda_, step=0.1)
    chi = st.sidebar.slider("Arbitrage Volatility (χ)", 0.005, 0.05, chi, step=0.005)
    alpha = st.sidebar.slider("Mispricing Impact (α)", 0.1, 1.0, alpha, step=0.1)
    eta_star = st.sidebar.slider("Mispricing Threshold (η*)", 0.01, 0.2, eta_star, step=0.01)
    kappa = st.sidebar.slider("Arbitrage Revival (κ)", 0.01, 0.3, kappa, step=0.01)
    rho_XY = st.sidebar.slider("Price-Volatility Correlation (ρ_XY)", -0.99, 0.99, rho_XY, step=0.01)
    rho_XZ = st.sidebar.slider("Price-Arbitrage Correlation (ρ_XZ)", -0.99, 0.99, rho_XZ, step=0.01)
    rho_YZ = st.sidebar.slider("Volatility-Arbitrage Correlation (ρ_YZ)", -0.99, 0.99, rho_YZ, step=0.01)

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
        spot_price = float(perp_ticker['mark_price']) if perp_ticker and perp_ticker.get('mark_price') else 111000
        forward_price = spot_price * np.exp(r_rate * ttm)
        atm_iv = df_options.iloc[(df_options['strike'] - forward_price).abs().argsort()[:1]]['iv'].iloc[0]
        if pd.isna(atm_iv) or atm_iv <= 0:
            atm_iv = np.sqrt(V0)
        S_u_orig = forward_price + 2 * forward_price * atm_iv * np.sqrt(ttm)
        S_l_orig = max(forward_price - 2 * forward_price * atm_iv * np.sqrt(ttm), 1e-6)

        with st.spinner("Simulating stochastic price paths..."):
            N = max(100, min(int(ttm * 365), 1000))
            S, V, eta = simulate_non_equilibrium(
                S0=spot_price, V0=V0, eta0=0.05, mu=mu, phi=phi, epsilon=epsilon, lambda_=lambda_,
                chi=chi, alpha=alpha, eta_star=eta_star, S_u=S_u_orig, S_l=S_l_orig, kappa=kappa,
                rho_XY=rho_XY, rho_XZ=rho_XZ, rho_YZ=rho_YZ, T=ttm, N=N, n_paths=2000
            )

        t_eval = np.linspace(0, ttm, N + 1)
        stochastic_df = pd.DataFrame({
            "Time": t_eval,
            "Price": np.mean(S, axis=0),
            "Volatility": np.sqrt(np.mean(V, axis=0)),
            "Arbitrage Return": np.mean(eta, axis=0),
            "Path": "Stochastic Mean",
            "ID": "Stochastic Mean"
        })

        # Volatility comparison
        realized_vol = np.std(log_returns) * np.sqrt(365) if len(log_returns) > 0 else np.sqrt(V0)
        st.subheader("Volatility Analysis")
        vol_cols = st.columns(2)
        vol_cols[0].metric("Implied Volatility (ATM)", f"{atm_iv:.2%}")
        vol_cols[1].metric("Realized Volatility (Historical)", f"{realized_vol:.2%}")
        if atm_iv > realized_vol * (1 + Z_SCORE_HIGH_IV_THRESHOLD):
            st.warning("Implied volatility significantly higher than realized: Potential overpricing or expected market stress.")
        elif atm_iv < realized_vol * (1 + Z_SCORE_LOW_IV_THRESHOLD):
            st.warning("Implied volatility significantly lower than realized: Potential underpricing or market stability.")

        # Derive cluster-based S_l and S_u
        final_prices = S[:, -1]
        final_prices = final_prices[np.isfinite(final_prices)]
        cluster_levels = []
        if len(final_prices) > 10:
            X = final_prices.reshape(-1, 1)
            db = DBSCAN(eps=spot_price * 0.05, min_samples=5).fit(X)
            labels = db.labels_
            unique_labels = set(labels) - {-1}
            if unique_labels:
                cluster_centers = [np.mean(final_prices[labels == label]) for label in unique_labels]
                if len(cluster_centers) >= 2:
                    cluster_centers.sort()
                    n_levels = min(4, len(cluster_centers))
                    indices = np.linspace(0, len(cluster_centers) - 1, n_levels, dtype=int)
                    cluster_levels = [cluster_centers[i] for i in indices]
                else:
                    cluster_levels = [np.percentile(final_prices, 2.5), np.percentile(final_prices, 97.5)]
            else:
                cluster_levels = [np.percentile(final_prices, 2.5), np.percentile(final_prices, 97.5)]
        else:
            cluster_levels = [S_l_orig, S_u_orig]
            logging.warning(f"Insufficient final prices for clustering: {len(final_prices)}. Using original S_l, S_u.")

        S_l = cluster_levels[0]
        S_u = cluster_levels[-1]

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

            # KDE with robust fallback
            final_prices = S[:, -1]
            final_prices = final_prices[np.isfinite(final_prices)]
            if len(final_prices) < 10:
                st.warning("Insufficient valid final prices for KDE. Using normal distribution fallback.")
                logging.warning(f"Final prices count: {len(final_prices)}. Using normal fallback.")
                mean_price = spot_price * np.exp(mu * ttm)
                std_price = spot_price * np.sqrt(V0) * np.sqrt(ttm) * 2.5
                final_prices = np.random.normal(mean_price, std_price, 1000)
                final_prices = final_prices[final_prices > 0]
            u = np.zeros_like(price_grid)
            if len(final_prices) > 0:
                try:
                    kde = gaussian_kde(final_prices)
                    u = kde(price_grid)
                except Exception as e:
                    st.error(f"KDE computation failed: {e}. Using uniform density.")
                    logging.error(f"KDE error: {e}")
                    u = np.ones_like(price_grid) / (price_grid[-1] - price_grid[0])
            else:
                st.error("No valid final prices. Using uniform density.")
                u = np.ones_like(price_grid) / (price_grid[-1] - price_grid[0])
            logging.info(f"KDE output - Mean density: {np.mean(u):.6f}, Valid prices: {len(final_prices)}")

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

                price_df = pd.DataFrame({"Time": times, "Price": prices, "Path": "Historical Price", "ID": "Historical"})
                stochastic_df['Time'] = t_eval
                sampled_indices = np.random.choice(2000, size=50, replace=False)
                sampled_paths = S[sampled_indices, :]
                path_dfs = [pd.DataFrame({"Time": t_eval, "Price": sampled_paths[i], "Path": "Simulated Path", "ID": str(i+1)}) for i in range(50)]
                combined_df = pd.concat([price_df, stochastic_df[['Time', 'Price', 'Path', 'ID']]] + path_dfs, ignore_index=True)

                max_time = max(times.max() if len(times) > 0 else 0, ttm)
                base = alt.Chart(combined_df).encode(
                    x=alt.X("Time:Q", title="Time (days)", scale=alt.Scale(domain=[0, max_time + 1])),
                    y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False, domain=[min(S_l_orig, S_l)-10000, max(S_u_orig, S_u)+10000])),
                    color=alt.Color("Path:N", scale=alt.Scale(domain=["Historical Price", "Stochastic Mean", "Simulated Path"], range=["#0000FF", "#FFA500", "#D3D3D3"])),
                    detail="ID:N"
                )
                historical_line = base.transform_filter(alt.datum.Path == "Historical Price").mark_line(strokeWidth=3)
                stochastic_line = base.transform_filter(alt.datum.Path == "Stochastic Mean").mark_line(strokeWidth=2)
                simulated_lines = base.transform_filter(alt.datum.Path == "Simulated Path").mark_line(strokeWidth=1)
                orig_support_df = pd.DataFrame({"Price": [S_l_orig]})
                orig_resistance_df = pd.DataFrame({"Price": [S_u_orig]})
                orig_support_lines = alt.Chart(orig_support_df).mark_rule(stroke="gray", strokeWidth=1, strokeDash=[4, 4]).encode(y="Price:Q")
                orig_resistance_lines = alt.Chart(orig_resistance_df).mark_rule(stroke="gray", strokeWidth=1, strokeDash=[4, 4]).encode(y="Price:Q")
                support_levels = [level for level in cluster_levels if level <= (S_l + S_u) / 2]
                resistance_levels = [level for level in cluster_levels if level > (S_l + S_u) / 2]
                support_df = pd.DataFrame({"Price": support_levels})
                resistance_df = pd.DataFrame({"Price": resistance_levels})
                support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
                resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                chart_layers = [historical_line, stochastic_line, simulated_lines, orig_support_lines, orig_resistance_lines, support_lines, resistance_lines]
                if pd.notna(lower_prob_range) and pd.notna(upper_prob_range):
                    prob_range_df = pd.DataFrame({
                        'lower_bound': [lower_prob_range],
                        'upper_bound': [upper_prob_range],
                        'label': [f"{confidence_level:.0%} Range"]
                    })
                    prob_band = alt.Chart(prob_range_df).mark_rect(opacity=0.1, color='yellow').encode(
                        y='lower_bound:Q',
                        y2='upper_bound:Q'
                    )
                    chart_layers.append(prob_band)
                else:
                    st.warning("Probability range not available due to invalid KDE output.")
                chart = alt.layer(*chart_layers).properties(
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
else:
    st.error("Could not load or process spot data. Check parameters or try again.")
