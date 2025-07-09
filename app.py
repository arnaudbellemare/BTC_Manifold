import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.euclidean
import geomstats
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
st.set_page_config(layout="wide", page_icon="📊", page_title="BTC Options and Manifold S/R Analysis")
st.title("BTC/USD Options and Volatility-Weighted Manifold Analysis")
st.markdown("""
This application combines options-based probability analysis with a volatility-weighted manifold model for BTC/USD.  
- **Options Analysis**: Implied volatility (IV) smiles and probability density functions (PDFs) using SVI model, ensuring no butterfly arbitrage. The PDF is used to calculate a central probability range (e.g., 68% confidence interval).
- **Manifold Analysis**: Models price-time space as a 2D manifold warped by options-implied volatility, with geodesic path (red) and S/R levels (green/red) on the PDF.  
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

# --- Geometric Modeling Class ---
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        self.dim = 2
        space = geomstats.geometry.euclidean.Euclidean(dim=self.dim)
        super().__init__(space=space)
        self.sigma = sigma
        self.t = t
        self.T = max(T, 1e-6)
        self.sigma_interp = interp1d(t, sigma, bounds_error=False, fill_value=(sigma[0], sigma[-1]))

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        sigma_val = max(self.sigma_interp(t_val), 1e-6)
        return np.diag([1.0, sigma_val**2]) + 1e-4 * np.eye(2)

    def christoffel_symbols(self, base_point):
        t_val = base_point[0]
        eps = 1e-6
        t_plus, t_minus = min(t_val + eps, self.T), max(t_val - eps, 0)
        sigma_val = max(self.sigma_interp(t_val), 1e-6)
        d_sigma_dt = (self.sigma_interp(t_plus) - self.sigma_interp(t_minus)) / (2 * eps)
        gamma = np.zeros((2, 2, 2))
        gamma[1, 0, 1] = (1 / sigma_val) * d_sigma_dt
        gamma[1, 1, 0] = gamma[1, 0, 1]
        gamma[0, 1, 1] = -sigma_val * d_sigma_dt
        return gamma

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
                time.sleep(3)  # Increased delay to avoid rate limits
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed to fetch Kraken data after {max_retries} attempts: {e}")
                    break
                time.sleep(2 ** (attempt + 2))  # Exponential backoff
    progress_bar.empty()
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
        if len(df) >= 10:
            return df
    st.error("Failed to fetch Kraken data. Using simulated data.")
    sim_t = pd.date_range(start=start_date, end=end_date, freq='h')
    n = len(sim_t)
    vol = np.random.normal(0, 0.02, n)
    vol = 0.01 + 0.005 * np.exp(-np.arange(n)/100) * np.cumsum(vol)
    sim_prices = 70000 * np.exp(np.cumsum(vol * np.random.normal(0, 1, n)))
    sim_df = pd.DataFrame({'datetime': sim_t, 'close': sim_prices, 'volume': np.random.randint(50, 200, n)})
    return sim_df

def geodesic_equation(s, y, metric_obj):
    pos, vel = y[:2], y[2:]
    gamma = metric_obj.christoffel_symbols(pos)
    accel = -np.einsum('ijk,j,k->i', gamma, vel, vel)
    return np.concatenate([vel, accel])

def get_hit_prob(level_list, price_grid, u, metric, T, epsilon, geodesic_prices):
    probs = []
    total_prob = np.trapz(u, price_grid)
    for level in level_list:
        mask = (price_grid >= level - epsilon) & (price_grid <= level + epsilon)
        raw_prob = np.trapz(u[mask], price_grid[mask])
        metric_mat = metric.metric_matrix([T, level])
        det = max(np.abs(np.linalg.det(metric_mat)), 1e-6)
        volume_element = np.sqrt(det)
        geodesic_price = np.interp(T, geodesic_prices["Time"], geodesic_prices["Price"])
        geodesic_weight = np.exp(-np.abs(level - geodesic_price) / (2 * epsilon))
        prob = raw_prob * volume_element * geodesic_weight
        probs.append(prob)
    total_level_prob = sum(probs) + 1e-10
    return [p / total_level_prob for p in probs] if total_level_prob > 0 else [0] * len(level_list)

def create_interactive_density_chart(price_grid, density, s_levels, r_levels, epsilon, forward_price, prob_range=None, confidence_level=None):
    if len(price_grid) == 0 or len(density) == 0:
        st.error("Density chart data is empty.")
        return None
    fig = go.Figure()

    # Add the probability range rectangle first, so it's in the background
    # --- FIX START ---
    # The original code had `all(pd.notna(prob_range))`, which fails because prob_range is a tuple.
    # The corrected code checks each element of the tuple individually.
    if prob_range and confidence_level is not None and pd.notna(prob_range[0]) and pd.notna(prob_range[1]):
        lower, upper = prob_range
        fig.add_vrect(x0=lower, x1=upper,
                      fillcolor="rgba(255, 255, 0, 0.2)", # A light yellow
                      layer="below", line_width=0,
                      annotation_text=f"{confidence_level:.0%} Prob. Range",
                      annotation_position="top left",
                      annotation_font_size=12)
    # --- FIX END ---

    fig.add_trace(go.Scatter(x=price_grid, y=density, mode='lines', name='Probability Density',
                            fill='tozeroy', line_color='lightblue', hovertemplate='Price: $%{x:,.2f}<br>Density: %{y:.4f}'))
    for level in s_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="green", opacity=0.2,
                      layer="below", line_width=0)
        fig.add_vline(x=level, line_color='green', line_dash='dash')
    for level in r_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="red", opacity=0.2,
                      layer="below", line_width=0)
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

def calculate_probability_range(price_grid, pdf_values, confidence_level):
    """
    Calculates the price range for a given confidence level from a PDF.
    Args:
        price_grid (np.ndarray): The grid of prices.
        pdf_values (np.ndarray): The probability density at each price.
        confidence_level (float): The desired confidence level (e.g., 0.68 for 68%).
    Returns:
        tuple: A tuple containing (lower_bound, upper_bound) of the price range.
               Returns (np.nan, np.nan) if calculation is not possible.
    """
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

# --- Thalex Options Helper Functions ---
@st.cache_data(ttl=300)
def get_thalex_instruments():
    url = f"{THALEX_BASE_URL}/public/instruments"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("result", [])
    except Exception as e:
        st.error(f"Failed to fetch Thalex instruments: {e}")
        logging.error(f"Thalex instruments fetch error: {e}")
        return []

def get_thalex_ticker(instrument_name: str):
    url = f"{THALEX_BASE_URL}/public/ticker"
    params = {"instrument_name": instrument_name}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("result", {})
    except Exception:
        return None

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

def simulate_options_data(coin: str) -> pd.DataFrame:
    strikes = np.linspace(60000, 100000, 20)
    types = ['C', 'P']
    sim_data = []
    for s in strikes:
        for t in types:
            iv = 0.5 + np.random.normal(0, 0.1)
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

# --- Black-Scholes Class ---
class BlackScholes:
    def __init__(self, time_to_maturity, strike, underlying_price, volatility, interest_rate):
        self.T = max(float(time_to_maturity), 1e-9)
        self.K = max(float(strike), 1e-9)
        self.S = max(float(underlying_price), 1e-9)
        self.sigma = max(float(volatility), 1e-9)
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

    def calculate_delta(self):
        if self.sigma_sqrt_T < 1e-9:
            call_delta = 1.0 if self.S > self.K else (0.0 if self.S < self.K else 0.5)
            put_delta = -1.0 if self.S < self.K else (0.0 if self.S > self.K else -0.5)
        else:
            call_delta = norm.cdf(self.d1)
            put_delta = norm.cdf(self.d1) - 1.0
        return call_delta, put_delta

    def calculate_gamma(self):
        if self.sigma_sqrt_T < 1e-9 or self.S < 1e-9:
            return 0.0
        return norm.pdf(self.d1) / (self.S * self.sigma_sqrt_T)

    def calculate_vega(self):
        if self.sigma_sqrt_T < 1e-9:
            return 0.0
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T) * 0.01

    def calculate_vanna(self):
        if self.sigma_sqrt_T < 1e-9 or self.sigma < 1e-9:
            return 0.0
        return -norm.pdf(self.d1) * self.d2 / self.sigma * 0.01

    def calculate_theta(self):
        if self.sigma_sqrt_T < 1e-9:
            call_theta_val = -self.r * self.K * np.exp(-self.r * self.T) if self.S < self.K else 0.0
            put_theta_val = -self.r * self.K * np.exp(-self.r * self.T) if self.S > self.K else 0.0
            return call_theta_val / 365.25, put_theta_val / 365.25
        term1_annual = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        call_theta_annual = term1_annual - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        put_theta_annual = term1_annual + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return call_theta_annual / 365.25, put_theta_annual / 365.25

# --- SVI Functions ---
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
        p0 = [atm_total_var_approx * 0.9, 0.1, -0.4, 0.0, 0.2]
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

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    if prices.empty or len(prices) < period + 1:
        return pd.Series(dtype='float64', index=prices.index)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = np.select(
        [loss == 0, gain == 0],
        [np.select([gain == 0], [50], default=100), 0],
        default=100 - (100 / (1 + rs))
    )
    return pd.Series(rsi, index=prices.index)

def calculate_realized_volatility(price_series: pd.Series, window: int, trading_periods_per_year: int = 365):
    if price_series.empty or len(price_series) < window + 1:
        return pd.Series(dtype='float64'), np.nan
    log_returns = np.log(price_series / price_series.shift(1))
    rv_series = log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(trading_periods_per_year)
    current_rv = rv_series.iloc[-1] if not rv_series.empty and pd.notna(rv_series.iloc[-1]) else np.nan
    return rv_series, current_rv
# --- Sidebar Configuration ---# --- Sidebar Configuration ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 90, 30)
epsilon_factor = st.sidebar.slider("S/R Zone Width Factor", 0.1, 2.0, 0.5, step=0.05, help="Controls the visual width of the Support/Resistance zones on the probability chart. Multiplied by price standard deviation.")
st.session_state['profitability_threshold'] = st.sidebar.slider("Profitability Confidence Interval (%)", 68, 99, 68, step=1) / 100.0
st.sidebar.header("Options Analysis Parameters")
all_instruments = get_thalex_instruments()
coin = "BTC"
expiries = get_expiries_from_instruments(all_instruments, coin)
sel_expiry = st.sidebar.selectbox("Options Expiry", expiries, index=0) if expiries else None
r_rate = st.sidebar.slider("Prime Rate (%)", 0.0, 14.0, 1.6, 0.1) / 100.0
use_oi_weights = st.sidebar.checkbox("Use OI Weights", value=True)
ivrv_n_days = st.sidebar.slider("N-day period for IV/RV analysis", 7, 180, 30, 1)
run_btn = st.sidebar.button("Run Analysis", use_container_width=True, type="primary", disabled=not sel_expiry)
if 'vol_regime_info' not in st.session_state:
    st.session_state.vol_regime_info = {
        "iv_rv_z_score": np.nan, "iv_rv_spread": np.nan, "current_atm_iv": np.nan,
        "current_n_day_rv": np.nan, "dtm_days_for_iv": np.nan, "n_day_for_rv": 30,
        "regime_category": "Normal", "rsi_14d": np.nan, "rsi_14h": np.nan,
        "rsi_14d_regime": "N/A", "rsi_14h_regime": "N/A", "combined_momentum_regime": "N/A"
    }

end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
with st.spinner("Fetching Kraken BTC/USD data..."):
    df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values
    times_pd = df['datetime']
    times = (times_pd - times_pd.iloc[0]).dt.total_seconds() / (24 * 3600)  # Convert to days
    T = times.iloc[-1] if not times.empty else 1.0
    N = len(prices)
    p0 = prices[0] if len(prices) > 0 else 70000
    returns = 100 * df['close'].pct_change().dropna()
    current_price = df['close'].iloc[-1] if not df['close'].empty else 70000
    if returns.empty:
        st.error("No valid returns data. Using default volatility.")
        sigma = np.full(N, 0.02)
    else:
        with st.spinner("Fitting GARCH model..."):
            try:
                model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
                sigma = model.conditional_volatility / 100
                sigma = np.pad(sigma, (N - len(sigma), 0), mode='edge')
                st.write(f"GARCH volatility range: {sigma.min():.6f} to {sigma.max():.6f}")
            except:
                st.warning("GARCH fitting failed. Using empirical volatility.")
                sigma = np.full(N, returns.std() / 100 if not returns.empty else 0.02)
                st.write(f"Empirical volatility: {sigma[0]:.6f}")
    col1, col2 = st.columns([2, 1])
    if sel_expiry and run_btn:
        st.header("Options-Based S/R Analysis (SVI)")
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
                atm_iv = 0.5
            sigma_options = np.full_like(times, atm_iv)
            metric = VolatilityMetric(sigma_options, times, ttm)
            # Compute geodesic path
            with col1:
                st.subheader("Price Path and Geodesic")
                with st.spinner("Computing geodesic path..."):
                    delta_p = prices[-1] - p0 if len(prices) > 1 else 0
                    recent_returns = returns[-min(24, len(returns)):].mean() / 100 if len(returns) > 1 else 0
                    y0 = np.concatenate([np.array([0.0, p0]), np.array([1.0, delta_p / T + recent_returns])])
                    t_eval = np.linspace(0, T, min(N, 100))
                    try:
                        sol = solve_ivp(geodesic_equation, [0, T], y0, args=(metric,), t_eval=t_eval, rtol=1e-5, method='Radau')
                        geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
                    except:
                        st.warning("Geodesic computation failed. Using linear path.")
                        linear_times = np.linspace(0, T, min(N, 100))
                        linear_prices = np.linspace(p0, current_price, min(N, 100))
                        volatility_adjustment = np.interp(linear_times, times, sigma)
                        adjusted_prices = linear_prices * (1 + 0.1 * volatility_adjustment)
                        geodesic_df = pd.DataFrame({"Time": linear_times, "Price": adjusted_prices, "Path": "Geodesic"})
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
                st.subheader("SVI Calibration Results")
                svi_cols = st.columns(5)
                svi_cols[0].metric("a", f"{svi_params['a']:.3f}")
                svi_cols[1].metric("b", f"{svi_params['b']:.3f}")
                svi_cols[2].metric("rho", f"{svi_params['rho']:.3f}")
                svi_cols[3].metric("m", f"{svi_params['m']:.3f}")
                svi_cols[4].metric("sigma", f"{svi_params['sigma']:.3f}")
                # Butterfly Arbitrage Check
                st.subheader("SVI Butterfly Arbitrage Check (g(k))")
                st.caption("For no butterfly arbitrage, g(k) must be >= 0.")
                min_k_check = np.log(max(1e-3, forward_price * 0.3) / forward_price)
                max_k_check = np.log(forward_price * 2.0 / forward_price)
                ks_to_check = np.linspace(min_k_check, max_k_check, 25)
                g_values = [check_svi_butterfly_arbitrage_gk(k_val, svi_params, ttm) for k_val in ks_to_check]
                df_g_check = pd.DataFrame({'Log-Moneyness (k)': ks_to_check, 'Strike (K)': forward_price * np.exp(ks_to_check), 'g(k)': g_values})
                st.dataframe(df_g_check.style.format({'Log-Moneyness (k)': "{:.3f}", 'Strike (K)': "${:,.0f}", 'g(k)':"{:.4f}"}).applymap(lambda val: 'color: red' if isinstance(val, (float,int)) and val < -1e-5 else '', subset=['g(k)']))
                if np.any(np.array(g_values) < -1e-5):
                    st.warning("SVI fit may contain butterfly arbitrage.")
                else:
                    st.success("SVI fit appears free of significant butterfly arbitrage at checked points.")
                # Generate PDF and S/R Levels
                price_grid = np.linspace(max(1, forward_price * 0.3), forward_price * 2.0, 300)
                log_moneyness = np.log(np.maximum(price_grid, 1e-6) / forward_price)
                svi_total_var = raw_svi_total_variance(log_moneyness, svi_params)
                svi_ivs = np.sqrt(np.maximum(svi_total_var, 1e-9) / ttm)
                call_prices_svi = np.array([BlackScholes(ttm, K, forward_price, iv, r_rate).calculate_prices()[0]
                                           for K, iv in zip(price_grid, svi_ivs)])
                pdf_df = get_pdf_from_svi_prices(price_grid, call_prices_svi, r_rate, ttm)
                u = pdf_df['pdf'].values
                # --- New Probability Range Calculation & Display ---
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
                        st.warning("Could not calculate the probability range for the selected confidence level.")
                # --- End of New Section ---
                price_std = forward_price * np.mean(svi_ivs) * np.sqrt(ttm)
                peak_height = np.percentile(u, 75)
                peak_distance = max(10, len(price_grid) // 50)
                peaks, _ = find_peaks(u, height=peak_height, distance=peak_distance)
                if len(peaks) < 4:
                    peaks, _ = find_peaks(u, height=0.01 * u.max(), distance=peak_distance // 2)
                levels = price_grid[peaks]
                warning_message = None
                if len(peaks) < 4:
                    warning_message = "Insufficient peaks detected. Using DBSCAN clustering."
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
                # Plot Price Path with S/R
                # Plot Price Path with S/R
                with col1:
                    if not geodesic_df.empty:
                        # Prepare historical price data for plotting
                        price_df = pd.DataFrame({
                            "Time": times,
                            "Price": prices,
                            "Path": "Historical Price"
                        })
                        # Combine historical and geodesic data
                        combined_df = pd.concat([price_df, geodesic_df], ignore_index=True)
                        # Create base chart with both historical price and geodesic path
                        base = alt.Chart(combined_df).encode(
                            x=alt.X("Time:Q", title="Time (days)"),
                            y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)),
                            color=alt.Color("Path:N", title="Path Type", scale=alt.Scale(domain=["Historical Price", "Geodesic"], range=["blue", "orange"]))
                        )
                        price_line = base.mark_line(strokeWidth=2).encode(detail='Path:N')
                        # Add S/R lines
                        support_df = pd.DataFrame({"Price": support_levels})
                        resistance_df = pd.DataFrame({"Price": resistance_levels})
                        support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
                        resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                        
                        # --- CHANGE START ---
                        # Create the chart layer for the probability range band
                        prob_band_chart = alt.Chart() # Start with an empty chart layer
                        
                        # Only add the band if the probability range is valid
                        if pd.notna(lower_prob_range) and pd.notna(upper_prob_range):
                            prob_range_df = pd.DataFrame([{
                                'lower_bound': lower_prob_range,
                                'upper_bound': upper_prob_range,
                                'label': f"{confidence_level:.0%} Range"
                            }])
                            
                            prob_band_chart = alt.Chart(prob_range_df).mark_rect(
                                opacity=0.1,
                                color='yellow'
                            ).encode(
                                y='lower_bound:Q',
                                y2='upper_bound:Q'
                            )

                        # Combine all layers, putting the probability band in the background first
                        chart = (prob_band_chart + price_line + support_lines + resistance_lines).properties(
                            title=f"Price Path, Geodesic, and {confidence_level:.0%} Probability Range",
                            height=500
                        ).interactive()
                        # --- CHANGE END ---
                        
                        try:
                            st.altair_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render price path chart: {e}")
                # Plot PDF with S/R
                with col2:
                    st.subheader("S/R Probabilities (SVI-Based)")
                    epsilon = epsilon_factor * price_std
                    if np.isnan(epsilon):
                        st.error("Invalid epsilon value for probability zones.")
                    else:
                        support_probs = get_hit_prob(support_levels, price_grid, u, metric, ttm, epsilon, geodesic_df)
                        resistance_probs = get_hit_prob(resistance_levels, price_grid, u, metric, ttm, epsilon, geodesic_df)
                        sub_col1, sub_col2 = st.columns(2)
                        with sub_col1:
                            st.markdown("**Support Levels**")
                            support_data = pd.DataFrame({'Level': support_levels, 'Hit %': support_probs})
                            if support_data.empty or support_data.isna().any().any():
                                st.warning("Support levels data is empty or contains NaN values.")
                            else:
                                st.dataframe(support_data.style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))

                        with sub_col2:
                            st.markdown("**Resistance Levels**")
                            resistance_data = pd.DataFrame({'Level': resistance_levels, 'Hit %': resistance_probs})
                            if resistance_data.empty or resistance_data.isna().any().any():
                                st.warning("Resistance levels data is empty or contains NaN values.")
                            else:
                                st.dataframe(resistance_data.style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))

                        with st.expander("Probability Distribution with S/R Zones", expanded=True):
                            st.markdown("""
                            Adjust the 'S/R Zone Width Factor' in the sidebar to control the visual width of S/R zones.  
                            The yellow shaded area shows the probability range based on the 'Profitability Confidence Interval' slider.
                            """)
                            interactive_density_fig = create_interactive_density_chart(
                                price_grid, u, support_levels, resistance_levels, epsilon, forward_price,
                                prob_range=(lower_prob_range, upper_prob_range),
                                confidence_level=confidence_level
                            )
                            if interactive_density_fig:
                                try:
                                    st.plotly_chart(interactive_density_fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Failed to render density chart: {e}")
                            if warning_message:
                                st.info(warning_message)
                # Volume Profile
                st.header("Historical Context: Volume-by-Price Analysis")
                st.markdown("""
                High-volume price levels act as strong support or resistance.  
                Green (support) and red (resistance) zones show SVI-derived S/R levels.  
                Orange dashed line: POC. Light blue solid line: Current price.
                """)
                volume_profile_fig, poc = create_volume_profile_chart(df, support_levels, resistance_levels, epsilon, current_price)
                if volume_profile_fig and poc is not None:
                    try:
                        st.plotly_chart(volume_profile_fig, use_container_width=True)
                        st.metric("Point of Control (POC)", f"${poc['price_bin']:,.2f}")
                        if current_price and not np.isnan(current_price):
                            st.metric("Current Price", f"${current_price:,.2f}")
                    except Exception as e:
                        st.error(f"Failed to render volume profile chart: {e}")
                # Options Chain
                st.header("Options Chain")
                df_options['model_iv'] = np.sqrt(np.maximum(raw_svi_total_variance(np.log(np.maximum(df_options['strike'], 1e-6)/forward_price), svi_params), 1e-9) / ttm)
                df_options['model_price'] = df_options.apply(
                    lambda r: BlackScholes(ttm, r['strike'], forward_price, r['model_iv'], r_rate).calculate_prices()[0 if r['type'] == 'C' else 1], axis=1)
                df_options['iv_diff'] = df_options['iv'] - df_options['model_iv']
                df_options['price_diff'] = df_options['mark_price'] - df_options['model_price']
                df_options['price_diff_pct'] = (df_options['price_diff'] / df_options['mark_price'].replace(0, np.nan)) * 100
                df_options['intrinsic_value'] = np.where(df_options['type'] == 'C',
                                                        np.maximum(0.0, spot_price - df_options['strike']),
                                                        np.maximum(0.0, df_options['strike'] - spot_price))
                df_options['time_value'] = np.maximum(0.0, df_options['mark_price'] - df_options['intrinsic_value'])
                df_options['cc_cost'] = np.where(df_options['type'] == 'C', spot_price - df_options['mark_price'], np.nan)
                df_options['cc'] = (df_options['type'] == 'C') & (df_options['mark_price'] / spot_price > 0.01)
                greeks_data = {'delta': [], 'gamma': [], 'vega': [], 'theta': [], 'vanna': [], 'theta_gamma_ratio': []}
                for _, row in df_options.iterrows():
                    bs = BlackScholes(ttm, row['strike'], spot_price, row['iv'], r_rate)
                    dc, dp = bs.calculate_delta()
                    greeks_data['delta'].append(dc if row['type'] == 'C' else dp)
                    gamma_val = bs.calculate_gamma()
                    greeks_data['gamma'].append(gamma_val)
                    greeks_data['vega'].append(bs.calculate_vega())
                    tc, tp = bs.calculate_theta()
                    theta_val = tc if row['type'] == 'C' else tp
                    greeks_data['theta'].append(theta_val)
                    greeks_data['vanna'].append(bs.calculate_vanna())
                    greeks_data['theta_gamma_ratio'].append(theta_val / (gamma_val + 1e-9))
                for greek_name, values in greeks_data.items():
                    df_options[greek_name] = values
                cols_to_display = ['instrument', 'strike', 'type', 'mark_price', 'cc_cost', 'time_value', 'cc', 'iv', 'delta', 'gamma', 'theta_gamma_ratio', 'vega', 'theta', 'vanna', 'model_iv', 'iv_diff', 'model_price', 'price_diff', 'price_diff_pct', 'open_interest']
                cols_existing = [c for c in cols_to_display if c in df_options.columns]
                style_formats = {
                    'strike': '{:,.0f}', 'mark_price': '${:,.4f}', 'cc_cost': '${:,.2f}', 'time_value': '${:,.4f}',
                    'iv': '{:.2%}', 'delta': '{:.3f}', 'gamma': '{:.8f}', 'theta_gamma_ratio': '{:.2f}',
                    'vega': '{:.2f}', 'theta': '{:.3f}', 'vanna': '{:.4f}', 'model_iv': '{:.2%}',
                    'iv_diff': '{:.2%}', 'model_price': '${:,.4f}', 'price_diff': '${:,.4f}', 'price_diff_pct': '{:.1f}%',
                    'open_interest': '{:,.1f}'
                }
                style_formats_applied = {k: v for k, v in style_formats.items() if k in cols_existing}
                styled_df = df_options[cols_existing].style.format(style_formats_applied).background_gradient(
                    subset=['iv_diff', 'price_diff_pct'], cmap='RdYlGn', vmin=-0.10, vmax=0.10, axis=0)
                st.dataframe(styled_df, height=400, use_container_width=True)
                # IV/RV Analysis
                st.header("Implied vs. Realized Volatility & Momentum Analysis")
                st.session_state.vol_regime_info['dtm_days_for_iv'] = ttm * 365.25
                current_atm_iv = df_options.iloc[(df_options['strike'] - forward_price).abs().argsort()[:1]]['iv'].iloc[0]
                if pd.isna(current_atm_iv) or current_atm_iv <= 0:
                    current_atm_iv = np.nan
                st.session_state.vol_regime_info['current_atm_iv'] = current_atm_iv
                ohlcv_rv_df = fetch_kraken_data('BTC/USD', '1d', end_date - pd.Timedelta(days=ivrv_n_days + 60), end_date)
                hist_rv_series, curr_n_day_rv = (pd.Series(dtype='float64'), np.nan)
                if not ohlcv_rv_df.empty and 'close' in ohlcv_rv_df.columns and len(ohlcv_rv_df['close'].dropna()) >= ivrv_n_days + 1:
                    hist_rv_series, curr_n_day_rv = calculate_realized_volatility(ohlcv_rv_df['close'], window=ivrv_n_days)
                st.session_state.vol_regime_info['current_n_day_rv'] = curr_n_day_rv if pd.notna(curr_n_day_rv) else np.nan
                st.session_state.vol_regime_info['n_day_for_rv'] = ivrv_n_days
                m_hist_rv, s_hist_rv, rv_z_calc = np.nan, np.nan, np.nan
                can_calc_rv_z = False
                if not hist_rv_series.empty and len(hist_rv_series.dropna()) > 10:
                    hrv_data = hist_rv_series.dropna()
                    if len(hrv_data) > 10:
                        m_hist_rv, s_hist_rv = hrv_data.mean(), hrv_data.std()
                        if pd.notna(current_atm_iv) and pd.notna(m_hist_rv) and pd.notna(s_hist_rv) and s_hist_rv > 1e-6:
                            rv_z_calc = (current_atm_iv - m_hist_rv) / s_hist_rv
                            can_calc_rv_z = True
                st.session_state.vol_regime_info['iv_rv_z_score'] = rv_z_calc if can_calc_rv_z else np.nan
                ivrv_spr = current_atm_iv - curr_n_day_rv if pd.notna(current_atm_iv) and pd.notna(curr_n_day_rv) else np.nan
                st.session_state.vol_regime_info['iv_rv_spread'] = ivrv_spr
                if can_calc_rv_z and pd.notna(rv_z_calc):
                    if rv_z_calc > Z_SCORE_HIGH_IV_THRESHOLD:
                        st.session_state.vol_regime_info['regime_category'] = "High IV"
                    elif rv_z_calc < Z_SCORE_LOW_IV_THRESHOLD:
                        st.session_state.vol_regime_info['regime_category'] = "Low IV"
                    else:
                        st.session_state.vol_regime_info['regime_category'] = "Normal IV"
                else:
                    st.session_state.vol_regime_info['regime_category'] = "Unknown (RV Z-score N/A)"
                iv_interpretation_text = [
                    f"Current ATM IV for this {ttm*365.25:.0f}-day expiry is **{current_atm_iv:.2%}**." if pd.notna(current_atm_iv) else "Current ATM IV: N/A."
                ]
                if can_calc_rv_z:
                    if rv_z_calc > 1.5:
                        iv_interpretation_text.append(f" This is **significantly above** historical average {ivrv_n_days}-D RV ({m_hist_rv:.2%}).")
                    elif rv_z_calc < -1.5:
                        iv_interpretation_text.append(f" This is **significantly below** historical average {ivrv_n_days}-D RV ({m_hist_rv:.2%}).")
                    else:
                        iv_interpretation_text.append(f" This is **near** historical average {ivrv_n_days}-D RV ({m_hist_rv:.2%}).")
                else:
                    iv_interpretation_text.append(" Z-score vs historical RV not available.")
                st.markdown(" ".join(iv_interpretation_text))
                st.metric("Current Volatility Regime (IV vs RV)", st.session_state.vol_regime_info['regime_category'])
                metric_cols = st.columns(4)
                metric_cols[0].metric(f"ATM IV ({ttm*365.25:.0f}D)", f"{current_atm_iv:.2%}" if pd.notna(current_atm_iv) else "N/A")
                metric_cols[1].metric(f"{ivrv_n_days}-D RV (Realized)", f"{curr_n_day_rv:.2%}" if pd.notna(curr_n_day_rv) else "N/A")
                metric_cols[2].metric("IV - RV Spread", f"{ivrv_spr:.2%}" if pd.notna(ivrv_spr) else "N/A")
                metric_cols[3].metric("IV vs Hist. RV Z-Score", f"{rv_z_calc:.2f}" if pd.notna(rv_z_calc) else "N/A")
                # RSI Analysis
                rsi_14d, rsi_14h = np.nan, np.nan
                ohlcv_d_rsi = fetch_kraken_data('BTC/USD', '1d', end_date - pd.Timedelta(days=60), end_date)
                if not ohlcv_d_rsi.empty and 'close' in ohlcv_d_rsi.columns and len(ohlcv_d_rsi['close'].dropna()) >= 15:
                    rsi_14d_s = calculate_rsi(ohlcv_d_rsi['close'].dropna(), period=14)
                    if not rsi_14d_s.empty and pd.notna(rsi_14d_s.iloc[-1]):
                        rsi_14d = rsi_14d_s.iloc[-1]
                ohlcv_h_rsi = fetch_kraken_data('BTC/USD', '1h', end_date - pd.Timedelta(days=14), end_date)
                if not ohlcv_h_rsi.empty and 'close' in ohlcv_h_rsi.columns and len(ohlcv_h_rsi['close'].dropna()) >= 15:
                    rsi_14h_s = calculate_rsi(ohlcv_h_rsi['close'].dropna(), period=14)
                    if not rsi_14h_s.empty and pd.notna(rsi_14h_s.iloc[-1]):
                        rsi_14h = rsi_14h_s.iloc[-1]
                def interpret_rsi_val(rsi_value, period_label):
                    if pd.isna(rsi_value):
                        return f"{period_label} RSI: N/A (Insufficient data)", "N/A"
                    if rsi_value >= RSI_BULLISH_THRESHOLD:
                        return f"{period_label} RSI: {rsi_value:.2f} (Bullish - Overbought or strong upward momentum)", "Bullish"
                    elif rsi_value <= RSI_BEARISH_THRESHOLD:
                        return f"{period_label} RSI: {rsi_value:.2f} (Bearish - Oversold or strong downward momentum)", "Bearish"
                    return f"{period_label} RSI: {rsi_value:.2f} (Neutral - No strong directional momentum)", "Neutral"
                rsi_14d_text, rsi_14d_cat = interpret_rsi_val(rsi_14d, "Daily (14D)")
                rsi_14h_text, rsi_14h_cat = interpret_rsi_val(rsi_14h, "Hourly (14H)")
                combined_momentum = "Strong Bullish Alignment" if rsi_14d_cat == "Bullish" and rsi_14h_cat == "Bullish" else \
                                    "Strong Bearish Alignment" if rsi_14d_cat == "Bearish" and rsi_14h_cat == "Bearish" else \
                                    "Neutral/Mixed Momentum"
                st.session_state.vol_regime_info.update({
                    'rsi_14d': rsi_14d, 'rsi_14h': rsi_14h, 'rsi_14d_regime': rsi_14d_text,
                    'rsi_14h_regime': rsi_14h_text, 'combined_momentum_regime': combined_momentum
                })
                rsi_cols = st.columns(3)
                rsi_cols[0].metric("RSI-14 Day", f"{rsi_14d:.2f}" if pd.notna(rsi_14d) else "N/A", help=rsi_14d_text.split('(')[0].strip())
                rsi_cols[1].metric("RSI-14 Hour", f"{rsi_14h:.2f}" if pd.notna(rsi_14h) else "N/A", help=rsi_14h_text.split('(')[0].strip())
                rsi_cols[2].metric("Combined Momentum", combined_momentum)
                st.caption(f"RSI Regimes: Bullish ≥ {RSI_BULLISH_THRESHOLD}, Bearish ≤ {RSI_BEARISH_THRESHOLD}, Neutral otherwise.")
                # IV Smile Plot
                st.header("Market & SVI Volatility Smile")
                one_std_dev_price_move = forward_price * atm_iv * np.sqrt(ttm)
                min_strike_2sd = forward_price - 2 * one_std_dev_price_move
                max_strike_2sd = forward_price + 2 * one_std_dev_price_move
                plot_min_strike = max(0.1, min_strike_2sd * 0.95)
                plot_max_strike = max_strike_2sd * 1.05
                st.info(f"Plotting smile data within ~2 Standard Deviations of Forward ({plot_min_strike:.0f} - {plot_max_strike:.0f}).")
                col_disp1, col_disp2, col_disp3, col_disp4 = st.columns(4)
                col_disp1.metric("Spot (S)", f"${spot_price:,.2f}")
                col_disp2.metric("Forward (F)", f"${forward_price:,.2f}")
                col_disp3.metric("DTE", f"{ttm*365.25:.2f}")
                col_disp4.metric(f"Market ATM IV (K≈F)", f"{atm_iv:.2%}" if pd.notna(atm_iv) else "N/A")
                df_filtered = df_options[(df_options['strike'] >= plot_min_strike) & (df_options['strike'] <= plot_max_strike)]
                fig_sml = go.Figure()
                calls_filtered = df_filtered[df_filtered['type'] == 'C']
                puts_filtered = df_filtered[df_filtered['type'] == 'P']
                if not calls_filtered.empty:
                    fig_sml.add_trace(go.Scatter(x=calls_filtered['strike'], y=calls_filtered['iv'], mode='markers+lines', name='Calls IV (Market)', marker=dict(color='deepskyblue', size=7)))
                if not puts_filtered.empty:
                    fig_sml.add_trace(go.Scatter(x=puts_filtered['strike'], y=puts_filtered['iv'], mode='markers+lines', name='Puts IV (Market)', marker=dict(color='salmon', size=7, symbol='x')))
                fig_sml.add_vline(x=spot_price, line_dash="dash", line_color="grey", annotation_text="Spot (S)")
                fig_sml.add_vline(x=forward_price, line_dash="dot", line_color="lightslategrey", annotation_text="Fwd (F)")
                fineK_grid = np.linspace(max(1e-3, plot_min_strike * 0.9), plot_max_strike * 1.1, 200)
                fine_log_moneyness = np.log(np.maximum(fineK_grid, 1e-6) / forward_price)
                svi_ivs_fine = np.sqrt(np.maximum(raw_svi_total_variance(fine_log_moneyness, svi_params), 1e-9) / ttm)
                fig_sml.add_trace(go.Scatter(x=fineK_grid, y=svi_ivs_fine, mode='lines', name='SVI Fit', line=dict(color='lightgreen', width=2, dash='dashdot')))
                fig_sml.update_layout(
                    title=f"Market and SVI IV Smile: {coin}-{sel_expiry}",
                    xaxis_title="Strike ($)",
                    yaxis_title="Implied Volatility",
                    yaxis_tickformat=".1%",
                    template="plotly_dark",
                    xaxis_range=[plot_min_strike, plot_max_strike]
                )
                st.plotly_chart(fig_sml, use_container_width=True)
            else:
                st.error("SVI calibration failed.")
                # Still display price path without S/R lines
                with col1:
                    if not geodesic_df.empty:
                        price_df = pd.DataFrame({
                            "Time": times,
                            "Price": prices,
                            "Path": "Historical Price"
                        })
                        combined_df = pd.concat([price_df, geodesic_df], ignore_index=True)
                        base = alt.Chart(combined_df).encode(
                            x=alt.X("Time:Q", title="Time (days)"),
                            y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)),
                            color=alt.Color("Path:N", title="Path Type", scale=alt.Scale(domain=["Historical Price", "Geodesic"], range=["blue", "red"]))
                        )
                        price_line = base.mark_line(strokeWidth=2).encode(detail='Path:N')
                        chart = price_line.properties(
                            title="Price Path and Geodesic (No S/R due to failed calibration)", height=500
                        ).interactive()
                        try:
                            st.altair_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render price path chart: {e}")
            # Export Results
            st.header("Export Results")
            export_button = st.button("Download Charts and Data")
            if export_button:
                with st.spinner("Generating exports..."):
                    try:
                        chart.save("price_chart.html")
                        with open("price_chart.html", "rb") as f:
                            st.download_button("Download Price Chart", f, file_name="price_chart.html")
                    except Exception as e:
                        st.warning(f"Failed to export price chart: {e}")
                    if 'interactive_density_fig' in locals() and interactive_density_fig:
                        interactive_density_fig.write_html("density_chart.html")
                        with open("density_chart.html", "rb") as f:
                            st.download_button("Download Density Chart", f, file_name="density_chart.html")
                    if 'volume_profile_fig' in locals() and volume_profile_fig:
                        volume_profile_fig.write_html("volume_profile.html")
                        with open("volume_profile.html", "rb") as f:
                            st.download_button("Download Volume Profile", f, file_name="volume_profile.html")
                    if svi_params:
                        sr_data = pd.concat([
                            pd.DataFrame({'Type': 'Support', 'Level': support_levels, 'Hit %': support_probs}),
                            pd.DataFrame({'Type': 'Resistance', 'Level': resistance_levels, 'Hit %': resistance_probs})
                        ])
                        st.download_button("Download S/R Data", sr_data.to_csv(index=False), file_name="sr_levels.csv")
                    if df_options is not None and not df_options.empty:
                        st.download_button("Download Options Data", df_options.to_csv(index=False), file_name="options_data.csv")
        else:
            st.error("No valid options data available.")
            # Display price path without S/R lines
            with col1:
                st.subheader("Price Path and Geodesic")
                with st.spinner("Computing geodesic path..."):
                    delta_p = prices[-1] - p0 if len(prices) > 1 else 0
                    recent_returns = returns[-min(24, len(returns)):].mean() / 100 if len(returns) > 1 else 0
                    y0 = np.concatenate([np.array([0.0, p0]), np.array([1.0, delta_p / T + recent_returns])])
                    t_eval = np.linspace(0, T, min(N, 100))
                    metric = VolatilityMetric(sigma, times, T)
                    try:
                        sol = solve_ivp(geodesic_equation, [0, T], y0, args=(metric,), t_eval=t_eval, rtol=1e-5, method='Radau')
                        geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
                    except:
                        st.warning("Geodesic computation failed. Using linear path.")
                        linear_times = np.linspace(0, T, min(N, 100))
                        linear_prices = np.linspace(p0, current_price, min(N, 100))
                        volatility_adjustment = np.interp(linear_times, times, sigma)
                        adjusted_prices = linear_prices * (1 + 0.1 * volatility_adjustment)
                        geodesic_df = pd.DataFrame({"Time": linear_times, "Price": adjusted_prices, "Path": "Geodesic"})
                if not geodesic_df.empty:
                    price_df = pd.DataFrame({
                        "Time": times,
                        "Price": prices,
                        "Path": "Historical Price"
                    })
                    combined_df = pd.concat([price_df, geodesic_df], ignore_index=True)
                    base = alt.Chart(combined_df).encode(
                        x=alt.X("Time:Q", title="Time (days)"),
                        y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)),
                        color=alt.Color("Path:N", title="Path Type", scale=alt.Scale(domain=["Historical Price", "Geodesic"], range=["blue", "red"]))
                    )
                    price_line = base.mark_line(strokeWidth=2).encode(detail='Path:N')
                    chart = price_line.properties(
                        title="Price Path and Geodesic (No S/R due to no options data)", height=500
                    ).interactive()
                    try:
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to render price path chart: {e}")
    else:
        st.info("Select an options expiry and click 'Run Analysis' to enable S/R analysis.")
else:
    st.error("Could not load or process spot data. Check parameters or try again.")
st.markdown("""
<style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)
