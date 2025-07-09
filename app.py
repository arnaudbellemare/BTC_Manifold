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
import warnings
import requests
from datetime import datetime, timezone, timedelta
import re
import logging
from typing import Optional, Dict, List
import time

# --- Global Settings ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
st.set_page_config(layout="wide")
st.title("BTC/USD Price and Options-Based S/R Analysis on a Volatility-Weighted Manifold")
st.markdown("""
This application models the Bitcoin market as a 2D geometric space (manifold) of (Time, Log-Price), warped by options-implied volatility, with support/resistance (S/R) levels derived from the SVI model.  
- **Geodesic (Red Line):** The "straightest" path through the volatility landscape.  
- **S/R Grid:** Support (green) and resistance (red) levels from the SVI implied probability density.  
- **Volume Profile:** Historical trading activity with S/R levels, POC (orange), and current price (light blue).  
- **Options Analysis:** Implied probability density from SVI model, with S/R levels overlaid.  
*Use the sidebar to adjust parameters. Hover over charts for details.*
""")

# --- Thalex API Configuration ---
THALEX_BASE_URL = "https://thalex.com/api/v2"
REQUEST_TIMEOUT = 15

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
    timeframe_seconds = 3600  # 1 hour
    all_ohlcv = []
    max_retries = 5
    progress_bar = st.progress(0)
    while since < int(end_date.timestamp() * 1000):
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=500)  # Reduced limit to avoid rate limits
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = int(ohlcv[-1][0]) + timeframe_seconds * 1000
                progress_bar.progress(min(1.0, since / (end_date.timestamp() * 1000)))
                time.sleep(2)  # Increased delay to respect rate limits
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed to fetch Kraken data after {max_retries} attempts: {e}")
                    break
                time.sleep(2 ** (attempt + 1))  # Exponential backoff
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

def create_interactive_density_chart(price_grid, density, s_levels, r_levels, epsilon):
    if len(price_grid) == 0 or len(density) == 0:
        st.error("Density chart data is empty.")
        return None
    fig = go.Figure()
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
    fig.update_layout(
        title="Options Implied Probability Distribution with S/R Zones",
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
        time.sleep(0.05)  # Increased delay for Thalex API
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

# --- SVI Functions ---
def raw_svi_total_variance(k, params):
    a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']
    b = max(0, b)
    rho = np.clip(rho, -0.9999, 0.9999)
    sigma = max(1e-5, sigma)
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

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
        logging.error(f"Error in SVI butterfly arbitrage check for k={k_val}")
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

def check_raw_svi_params_validity(params):
    a, b, rho, sigma = params['a'], params.get('b', 0), params.get('rho', 0), params.get('sigma', 1e-3)
    if b < 0 or not (-1 < rho < 1) or sigma <= 0:
        return False
    return (a + b * sigma * np.sqrt(1 - rho**2)) >= -1e-9

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
        logging.info(f"SVI Calibration Success: {result.message}, Error: {result.fun:.2e}, Params: {cal_p}")
        return cal_p, result.fun
    logging.error(f"SVI Calibration failed: {result.message}")
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
    integral_pdf = np.trapezoid(pdf_values, K_grid_dense)
    if integral_pdf > 1e-6:
        pdf_values /= integral_pdf
    return pd.DataFrame({'strike': K_grid_dense, 'pdf': pdf_values})

# --- Main Application Logic ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 90, 30)
epsilon_factor = st.sidebar.slider("Probability Range Factor", 0.1, 2.0, 0.5, step=0.05)
st.sidebar.header("Options Analysis Parameters")
all_instruments = get_thalex_instruments()
coin = "BTC"
expiries = get_expiries_from_instruments(all_instruments, coin)
sel_expiry = st.sidebar.selectbox("Options Expiry", expiries, index=0) if expiries else None
r_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 1.6, 0.1) / 100.0
use_oi_weights = st.sidebar.checkbox("Use OI Weights", value=True)
reset_button = st.sidebar.button("Reset to Defaults")
if reset_button:
    days_history, epsilon_factor = 30, 0.5

end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
with st.spinner("Fetching Kraken BTC/USD data..."):
    df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values
    times_pd = df['datetime']
    times = (times_pd - times_pd.iloc[0]).dt.total_seconds() / (24 * 3600)  # Convert to days
    T = times.iloc[-1] if not times.empty else 1.0  # Fallback to 1 day
    N = len(prices)
    p0 = prices[0] if len(prices) > 0 else 70000  # Fallback price
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
    # --- Options Analysis ---
    if sel_expiry:
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
            # Use ATM IV from options data as volatility for manifold
            atm_iv = df_options.iloc[(df_options['strike'] - forward_price).abs().argsort()[:1]]['iv'].iloc[0]
            if pd.isna(atm_iv) or atm_iv <= 0:
                atm_iv = 0.5  # Fallback
            sigma_options = np.full_like(times, atm_iv)  # Constant IV over TTM
            metric = VolatilityMetric(sigma_options, times, ttm)
            with col1:
                if not geodesic_df.empty:
                    base = alt.Chart(geodesic_df).encode(
                        x=alt.X("Time:Q", title="Time (days)"),
                        y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False))
                    )
                    geodesic_line = base.mark_line(strokeWidth=3, color="red").encode(detail='Path:N')
                    # S/R levels will be added after computing from SVI
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
                # Generate PDF
                price_grid = np.linspace(max(1, forward_price * 0.3), forward_price * 2.0, 300)
                log_moneyness = np.log(np.maximum(price_grid, 1e-6) / forward_price)
                svi_total_var = raw_svi_total_variance(log_moneyness, svi_params)
                svi_ivs = np.sqrt(np.maximum(svi_total_var, 1e-9) / ttm)
                call_prices_svi = np.array([BlackScholes(ttm, K, forward_price, iv, r_rate).calculate_prices()[0]
                                           for K, iv in zip(price_grid, svi_ivs)])
                pdf_df = get_pdf_from_svi_prices(price_grid, call_prices_svi, r_rate, ttm)
                # Compute S/R levels
                price_std = forward_price * np.mean(svi_ivs) * np.sqrt(ttm)
                u = pdf_df['pdf'].values
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

                        with st.expander("Tune Probability Range Factor", expanded=True):
                            st.markdown("""
                            Adjust the 'Probability Range Factor' to control S/R zone width.  
                            - Smaller values: tighter zones.  
                            - Larger values: broader zones.  
                            **Recommended: 0.3–0.7.**
                            """)
                            interactive_density_fig = create_interactive_density_chart(price_grid, u, support_levels, resistance_levels, epsilon)
                            if interactive_density_fig:
                                try:
                                    st.plotly_chart(interactive_density_fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Failed to render density chart: {e}")
                            if warning_message:
                                st.info(warning_message)
                # Add S/R to price chart
                with col1:
                    support_df = pd.DataFrame({"Price": support_levels})
                    resistance_df = pd.DataFrame({"Price": resistance_levels})
                    support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
                    resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                    chart = (geodesic_line + support_lines + resistance_lines).properties(
                        title="Price Path, Geodesic, and S/R Grid", height=500
                    ).interactive()
                    try:
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to render price path chart: {e}")
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
                st.subheader("Options Chain")
                st.dataframe(df_options[['instrument', 'strike', 'type', 'mark_price', 'iv', 'delta']].style.format({
                    'strike': '{:,.0f}',
                    'mark_price': '${:.2f}',
                    'iv': '{:.2%}',
                    'delta': '{:.2f}'
                }))
            else:
                st.error("SVI calibration failed.")
        else:
            st.error("No valid options data available.")
    else:
        st.info("Select an options expiry to enable S/R analysis.")

    # --- Export Results ---
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
            if interactive_density_fig:
                interactive_density_fig.write_html("density_chart.html")
                with open("density_chart.html", "rb") as f:
                    st.download_button("Download Density Chart", f, file_name="density_chart.html")
            if volume_profile_fig:
                volume_profile_fig.write_html("volume_profile.html")
                with open("volume_profile.html", "rb") as f:
                    st.download_button("Download Volume Profile", f, file_name="volume_profile.html")
            sr_data = pd.concat([
                pd.DataFrame({'Type': 'Support', 'Level': support_levels, 'Hit %': support_probs}),
                pd.DataFrame({'Type': 'Resistance', 'Level': resistance_levels, 'Hit %': resistance_probs})
            ])
            st.download_button("Download S/R Data", sr_data.to_csv(index=False), file_name="sr_levels.csv")
            if df_options is not None and not df_options.empty:
                st.download_button("Download Options Data", df_options.to_csv(index=False), file_name="options_data.csv")

else:
    st.error("Could not load or process spot data. Check parameters or try again.")

# --- Custom CSS ---
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
