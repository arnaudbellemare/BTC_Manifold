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
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import warnings
from joblib import Parallel, delayed
import streamlit.components.v1 as components
import requests
from datetime import datetime, timezone, timedelta
import re
import logging
from typing import Optional, Dict, List
from scipy.stats import norm
from scipy.optimize import minimize, brentq

# --- Global Settings ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
st.set_page_config(layout="wide")
st.title("BTC/USD Price and Options Analysis on a Volatility-Weighted Manifold")
st.markdown("""
This application models the Bitcoin market as a 2D geometric space (manifold) of (Time, Log-Price), warped by GARCH-derived volatility, and integrates options-based probability analysis.  
- **Geodesic (Red Line):** The "straightest" path through the volatility landscape.  
- **S/R Grid:** Support (green) and resistance (red) levels from Monte Carlo simulations.  
- **Volume Profile:** Historical trading activity with S/R levels, POC (orange), and current price (light blue).  
- **Options Analysis:** Implied probability density from SABR model and top trade ideas based on mispricing.  
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
    timeframe_seconds = 3600
    all_ohlcv = []
    max_retries = 5
    progress_bar = st.progress(0)
    while since < int(end_date.timestamp() * 1000):
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=720)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = int(ohlcv[-1][0]) + timeframe_seconds * 1000
                progress_bar.progress(min(1.0, since / (end_date.timestamp() * 1000)))
                time.sleep(2 ** attempt)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed to fetch Kraken data after {max_retries} attempts: {e}")
                    break
                continue
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

def simulate_single_path(p0, mu, sigma, T, N, dt, seed):
    np.random.seed(seed)
    path = np.zeros(N)
    path[0] = p0
    dW = np.random.normal(0, np.sqrt(dt), N - 1)
    for j in range(N - 1):
        path[j + 1] = path[j] * np.exp((mu - 0.5 * sigma[j]**2) * dt + sigma[j] * dW[j])
    return path

@st.cache_data
def simulate_paths(p0, mu, sigma, T, N, n_paths):
    if N < 2:
        return np.array([[p0]] * n_paths), np.array([0])
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = Parallel(n_jobs=-1)(delayed(simulate_single_path)(p0, mu, sigma, T, N, dt, i) for i in range(n_paths))
    return np.array(paths), t

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

def manifold_distance(x, y, metric, T):
    point_x = np.array([T, x])
    point_y = np.array([T, y])
    metric_mat = metric.metric_matrix([T, (x + y) / 2])
    delta = point_x - point_y
    return np.sqrt(max(delta.T @ metric_mat @ delta, 1e-6))

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
        title="Probability Distribution with S/R Zones",
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
        time.sleep(0.02)
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
            if self.S > self.K: self.d1, self.d2 = np.inf, np.inf
            elif self.S < self.K: self.d1, self.d2 = -np.inf, -np.inf
            else: self.d1, self.d2 = 0, 0
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

# --- Arbitrage-Free SABR Pricer ---
@st.cache_data(ttl=300)
def arbitrage_free_sabr_pricer(alpha: float, beta: float, rho: float, nu: float, F: float, T: float, K_vals: np.ndarray, r_rate: float = 0.0) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float, float]:
    N_time_steps = 50
    N_cells = 400
    N_nodes = N_cells + 1
    if N_cells < 2:
        logging.error("N_cells must be at least 2 for PDE scheme.")
        nan_prices = np.full(len(K_vals), np.nan)
        return nan_prices, nan_prices, pd.DataFrame({'strike': [], 'pdf': []}), 0.0, 0.0
    dt = T / N_time_steps
    vol_approx = alpha * (F**(beta-1)) if beta < 1 and F > 0 else alpha
    sigma_sqrt_T_approx = vol_approx * np.sqrt(T)
    F_min_domain = max(0.001, F * np.exp(-7 * sigma_sqrt_T_approx - 0.5 * vol_approx**2 * T))
    F_max_domain = F * np.exp(7 * sigma_sqrt_T_approx - 0.5 * vol_approx**2 * T)
    if F_max_domain <= F_min_domain + 0.1:
        F_max_domain = F_min_domain + max(20, F*2.0)
    h_cell_width = (F_max_domain - F_min_domain) / N_cells
    F_comp_nodes = F_min_domain + (np.arange(N_cells) + 0.5) * h_cell_width
    if h_cell_width <= 1e-9:
        logging.error(f"h_cell_width too small ({h_cell_width}).")
        nan_prices = np.full(len(K_vals), np.nan)
        return nan_prices, nan_prices, pd.DataFrame({'strike': [], 'pdf': []}), 0.0, 0.0
    C_func = lambda f_val: f_val**beta if f_val > 1e-9 else (0.0 if beta > 1e-9 else 1e-9)
    if abs(beta - 1.0) < 1e-6:
        z_func = lambda f_val_loop: (nu / (alpha + 1e-9)) * (np.log(f_val_loop / (F + 1e-9))) if f_val_loop > 1e-9 and F > 1e-9 else 0.0
    else:
        z_func = lambda f_val_loop: (nu / (alpha + 1e-9)) * ((f_val_loop**(1.0-beta) - F**(1.0-beta)) / (1.0-beta)) if f_val_loop > 1e-9 and F > 1e-9 else 0.0
    def Gamma_func(f_val_loop):
        if abs(f_val_loop - F) < 1e-9:
            return beta * F**(beta - 1.0) if F > 1e-9 else 0.0
        return (C_func(f_val_loop) - C_func(F)) / (f_val_loop - F)
    Qc = np.zeros(N_cells)
    QL, QR = 0.0, 0.0
    f_idx_initial = np.searchsorted(F_comp_nodes - h_cell_width/2, F)
    f_idx_clamped = np.clip(f_idx_initial, 0, N_cells - 1)
    Qc[f_idx_clamped] = 1.0 / h_cell_width
    A_tridiag = np.zeros((3, N_cells))
    for i_time_step in range(1, N_time_steps + 1):
        current_t = i_time_step * dt
        Qc_prev_step = Qc.copy()
        _C_vals = np.array([C_func(f_val) for f_val in F_comp_nodes])
        _z_vals = np.array([z_func(f_val) for f_val in F_comp_nodes])
        _Gamma_vals = np.array([Gamma_func(f_val) for f_val in F_comp_nodes])
        D_sq_coeff_at_comp_nodes_NEW_T = (alpha**2) * (1 + 2*rho*nu*_z_vals + (nu**2)*(_z_vals**2)) * np.exp(rho*nu*_Gamma_vals*current_t) * (_C_vals**2)
        D_sq_coeff_at_comp_nodes_NEW_T = np.maximum(D_sq_coeff_at_comp_nodes_NEW_T, 1e-12)
        M_paper_NEW_T = 0.5 * D_sq_coeff_at_comp_nodes_NEW_T
        D_sq_coeff_at_comp_nodes_OLD_T = (alpha**2) * (1 + 2*rho*nu*_z_vals + (nu**2)*(_z_vals**2)) * np.exp(rho*nu*_Gamma_vals*(current_t - dt)) * (_C_vals**2)
        D_sq_coeff_at_comp_nodes_OLD_T = np.maximum(D_sq_coeff_at_comp_nodes_OLD_T, 1e-12)
        M_paper_OLD_T = 0.5 * D_sq_coeff_at_comp_nodes_OLD_T
        A_tridiag[1, :] = 1.0 + (dt / h_cell_width**2) * M_paper_NEW_T
        A_tridiag[0, 1:] = -(dt / (2 * h_cell_width**2)) * M_paper_NEW_T[:-1]
        A_tridiag[2, :-1] = -(dt / (2 * h_cell_width**2)) * M_paper_NEW_T[1:]
        RHS_vector = np.zeros(N_cells)
        RHS_vector[1:-1] = Qc_prev_step[1:-1] + (dt / (2 * h_cell_width**2)) * (
            M_paper_OLD_T[2:] * Qc_prev_step[2:] - 2*M_paper_OLD_T[1:-1] * Qc_prev_step[1:-1] + M_paper_OLD_T[:-2] * Qc_prev_step[:-2]
        )
        if N_cells >= 1:
            rhs_Q0_term = M_paper_OLD_T[1]*Qc_prev_step[1] - 2*M_paper_OLD_T[0]*Qc_prev_step[0]
            if M_paper_OLD_T[0] > 1e-9:
                rhs_Q0_term += M_paper_OLD_T[0]*(-Qc_prev_step[0])
            RHS_vector[0] = Qc_prev_step[0] + (dt / (2 * h_cell_width**2)) * rhs_Q0_term
            if M_paper_NEW_T[0] > 1e-9:
                A_tridiag[1,0] = 1.0 + (dt / h_cell_width**2) * M_paper_NEW_T[0] * (1.0 + 0.5)
                A_tridiag[0,1] = -(dt / (2 * h_cell_width**2)) * M_paper_NEW_T[0]
        if N_cells >= 2:
            idx_J = N_cells - 1
            rhs_QJ_term = M_paper_OLD_T[idx_J-1]*Qc_prev_step[idx_J-1] - 2*M_paper_OLD_T[idx_J]*Qc_prev_step[idx_J]
            if M_paper_OLD_T[idx_J] > 1e-9:
                rhs_QJ_term += M_paper_OLD_T[idx_J]*(-Qc_prev_step[idx_J])
            RHS_vector[idx_J] = Qc_prev_step[idx_J] + (dt / (2 * h_cell_width**2)) * rhs_QJ_term
            if M_paper_NEW_T[idx_J] > 1e-9:
                A_tridiag[1,idx_J] = 1.0 + (dt / h_cell_width**2) * M_paper_NEW_T[idx_J] * (1.0 + 0.5)
                A_tridiag[2,idx_J-1] = -(dt / (2 * h_cell_width**2)) * M_paper_NEW_T[idx_J]
        try:
            Qc = np.maximum(solve_banded((1, 1), A_tridiag, RHS_vector), 0)
        except:
            A_matrix_full_np = np.diag(A_tridiag[1,:]) + np.diag(A_tridiag[0,1:],1) + np.diag(A_tridiag[2,:-1],-1)
            try:
                Qc = np.maximum(np.linalg.solve(A_matrix_full_np, RHS_vector), 0)
            except:
                Qc = np.maximum(np.linalg.pinv(A_matrix_full_np) @ RHS_vector, 0)
        current_total_prob_on_grid = np.sum(Qc * h_cell_width)
        if current_total_prob_on_grid > 1e-9:
            Qc = Qc / current_total_prob_on_grid
    final_pdf_df = pd.DataFrame({'strike': F_comp_nodes, 'pdf': Qc})
    call_prices = np.zeros_like(K_vals, dtype=float)
    put_prices = np.zeros_like(K_vals, dtype=float)
    for i_k, K_strike in enumerate(K_vals):
        integrand_call_c = np.maximum(0, F_comp_nodes - K_strike) * Qc
        call_price_c = np.trapezoid(integrand_call_c, F_comp_nodes)
        integrand_put_c = np.maximum(0, K_strike - F_comp_nodes) * Qc
        put_price_c = np.trapezoid(integrand_put_c, F_comp_nodes)
        call_price_from_QR = max(0, F_max_domain - K_strike) * QR if K_strike < F_max_domain else 0
        put_price_from_QL = max(0, K_strike - F_min_domain) * QL if K_strike > F_min_domain else 0
        call_prices[i_k] = call_price_c + call_price_from_QR
        put_prices[i_k] = put_price_c + put_price_from_QL
    return call_prices, put_prices, final_pdf_df, QL, QR

# --- SABR Calibration ---
@st.cache_data(ttl=120)
def calibrate_arbitrage_free_sabr(df_market: pd.DataFrame, F: float, T: float, beta: float, sr_pct: float, use_oi_w: bool, r_rate: float) -> Tuple[Optional[float], Optional[float], Optional[float], float, int]:
    min_K_calib = F * (1 - sr_pct / 100.0)
    max_K_calib = F * (1 + sr_pct / 100.0)
    df_c = df_market[(df_market['strike'] >= min_K_calib) & (df_market['strike'] <= max_K_calib)].copy()
    if df_c.empty:
        st.warning(f"No strikes for SABR calibration in {sr_pct}% range of Forward price {F:.2f}.")
        return None, None, None, np.inf, 0
    K_for_calib = df_c['strike'].values
    P_market = df_c['mark_price'].values
    types_for_calib = df_c['type'].values
    OI_for_calib = df_c['open_interest'].values
    nPts = len(K_for_calib)
    if nPts < 3:
        st.warning(f"Need at least 3 unique strikes for SABR calibration ({nPts} found).")
        return None, None, None, np.inf, nPts
    w_calib = OI_for_calib / np.sum(OI_for_calib) if use_oi_w and np.sum(OI_for_calib) > 0 else np.ones(nPts) / nPts
    if use_oi_w and np.sum(OI_for_calib) > 0:
        avg_w = 1.0 / nPts
        w_calib = np.minimum(w_calib, 5 * avg_w)
        w_calib /= np.sum(w_calib)
    iteration_count = 0
    def sabr_calibration_objective_af(p_params):
        nonlocal iteration_count
        iteration_count += 1
        alpha_cal, rho_cal, nu_cal = p_params
        call_prices_model, put_prices_model, _unused_pdf, _unused_ql, _unused_qr = arbitrage_free_sabr_pricer(
            alpha_cal, beta, rho_cal, nu_cal, F, T, K_for_calib, r_rate
        )
        P_model = np.where(types_for_calib == 'C', call_prices_model, put_prices_model)
        price_diff = P_model - P_market
        penalty = np.sum(np.where(np.abs(price_diff) > 100, 1000 * (np.abs(price_diff) - 100), 0))
        error = np.sum(w_calib * (price_diff ** 2)) + penalty
        logging.debug(f"Iter {iteration_count}: alpha={alpha_cal:.4f}, rho={rho_cal:.4f}, nu={nu_cal:.4f}, Error={error:.6e}")
        return error
    atm_iv_market_approx = 0.5
    if 'iv' in df_market.columns and not df_market['iv'].empty:
        df_strikes_for_atm_iv = df_market[(df_market['strike'].isin(K_for_calib)) & (df_market['iv'].notna()) & (df_market['iv'] > 0)]
        if not df_strikes_for_atm_iv.empty:
            df_strikes_for_atm_iv = df_strikes_for_atm_iv.sort_values('strike')
            atm_iv_interp = np.interp(F, df_strikes_for_atm_iv['strike'], df_strikes_for_atm_iv['iv'],
                                    left=df_strikes_for_atm_iv['iv'].iloc[0],
                                    right=df_strikes_for_atm_iv['iv'].iloc[-1])
            if pd.notna(atm_iv_interp) and atm_iv_interp > 0:
                atm_iv_market_approx = atm_iv_interp
    wing_iv_low = df_market[df_market['strike'] < F]['iv'].mean() if not df_market[df_market['strike'] < F].empty else atm_iv_market_approx
    wing_iv_high = df_market[df_market['strike'] > F]['iv'].mean() if not df_market[df_market['strike'] > F].empty else atm_iv_market_approx
    initial_alpha = max(0.01, atm_iv_market_approx * 1.3 * (1 + 0.5 * (wing_iv_low - wing_iv_high) / atm_iv_market_approx))
    initial_guess_sabr = [initial_alpha, 0.2, 0.4]
    bounds_sabr = [(0.05, 1.5), (-0.8, 0.8), (0.05, 1.5)]
    try:
        logging.info(f"Starting Arbitrage-Free SABR calibration with F={F:.2f}, T={T:.4f}, beta={beta:.2f}, sr_pct={sr_pct}%.")
        res_sabr = minimize(
            sabr_calibration_objective_af, initial_guess_sabr,
            method='L-BFGS-B', bounds=bounds_sabr,
            options={'ftol': 1e-9, 'gtol': 1e-7, 'maxiter': 1000, 'eps': 1e-8}
        )
        if res_sabr.success or res_sabr.status in [0, 1, 2]:
            alpha_opt, rho_opt, nu_opt = res_sabr.x
            alpha_opt = max(bounds_sabr[0][0], min(alpha_opt, bounds_sabr[0][1]))
            rho_opt = max(bounds_sabr[1][0], min(rho_opt, bounds_sabr[1][1]))
            nu_opt = max(bounds_sabr[2][0], min(nu_opt, bounds_sabr[2][1]))
            logging.info(f"Arbitrage-Free SABR Calib Success: alpha={alpha_opt:.3f}, rho={rho_opt:.3f}, nu={nu_opt:.3f}, Error={res_sabr.fun:.2e}")
            return alpha_opt, rho_opt, nu_opt, res_sabr.fun, nPts
        else:
            st.error(f"Arbitrage-Free SABR Calibration failed: {res_sabr.message}")
            return None, None, None, np.inf, nPts
    except Exception as e:
        st.error(f"Arbitrage-Free SABR Calibration Exception: {e}")
        logging.error(f"Arbitrage-Free SABR Calib Exception: {e}", exc_info=True)
        return None, None, None, np.inf, nPts

# --- Trade Ideas Function ---
def generate_trade_ideas(df_options: pd.DataFrame, spot_price: float, top_n: int = 3) -> pd.DataFrame:
    if df_options.empty or not all(c in df_options.columns for c in ['strike', 'type', 'mark_price', 'delta', 'svi_dollar_mispricing']):
        return pd.DataFrame()
    df = df_options.copy()
    df['score'] = df.apply(
        lambda row: (row['svi_dollar_mispricing'] / row['mark_price'] if row['mark_price'] > 0 else 0) + (1 - abs(row['delta'])),
        axis=1
    )
    short_calls = df[(df['type'] == 'C') & (df['strike'] > spot_price) & (df['score'] > 0)].sort_values('score', ascending=False).head(top_n)
    short_puts = df[(df['type'] == 'P') & (df['strike'] < spot_price) & (df['score'] > 0)].sort_values('score', ascending=False).head(top_n)
    if short_calls.empty and short_puts.empty:
        return pd.DataFrame()
    trades = pd.concat([short_calls, short_puts])
    trades['candidate_type'] = np.where(trades['type'] == 'C', 'Short Call', 'Short Put')
    trades['display_name'] = trades.apply(lambda r: f"{r['type']} K={r['strike']:,.0f}", axis=1)
    trades['details'] = trades.apply(
        lambda r: f"Premium: ${r['mark_price']:.2f}, Misprice: ${r['svi_dollar_mispricing']:.2f}, Delta: {r['delta']:.2f}",
        axis=1
    )
    trades['rank'] = trades.groupby('candidate_type')['score'].rank(ascending=False, method='first').astype(int)
    return trades

# --- Main Application Logic ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 90, 30)
n_paths = st.sidebar.slider("Simulated Paths", 500, 5000, 2000, step=100)
n_display_paths = st.sidebar.slider("Displayed Paths", 10, 100, 50, step=10)
epsilon_factor = st.sidebar.slider("Probability Range Factor", 0.1, 2.0, 0.5, step=0.05)
st.sidebar.header("Options Analysis Parameters")
all_instruments = get_thalex_instruments()
coin = "BTC"
expiries = get_expiries_from_instruments(all_instruments, coin)
sel_expiry = st.sidebar.selectbox("Options Expiry", expiries, index=0) if expiries else None
r_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 1.6, 0.1) / 100.0
beta_sabr = st.sidebar.slider("SABR Beta", 0.0, 1.0, 0.9, 0.01)
sr_pct = st.sidebar.slider("SABR Strike Range (%)", 5, 100, 40, 1)
use_oi_weights = st.sidebar.checkbox("Use OI Weights", value=True)
reset_button = st.sidebar.button("Reset to Defaults")
if reset_button:
    days_history, n_paths, n_display_paths, epsilon_factor = 30, 2000, 50, 0.5

end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
with st.spinner("Fetching Kraken BTC/USD data..."):
    df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values
    times_pd = df['datetime']
    times = (times_pd - times_pd.iloc[0]).dt.total_seconds() / 3600
    T = times.iloc[-1] if not times.empty else 0
    N = len(prices)
    p0 = prices[0]
    returns = 100 * df['close'].pct_change().dropna()
    current_price = df['close'].iloc[-1] if not df['close'].empty else None
    if returns.empty:
        st.error("No valid returns data.")
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
        mu = returns.mean() / 100 if not returns.empty else 0
        with st.spinner("Simulating price paths..."):
            paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Price Paths and Geodesic")
            final_prices = paths[:, -1]
            price_std = np.std(final_prices)
            price_mean = np.mean(final_prices)
            if np.isnan(price_std) or np.isnan(price_mean):
                st.error("Invalid price statistics.")
            else:
                kde = gaussian_kde(final_prices, bw_method='scott')
                kde.set_bandwidth(bw_method=kde.factor * (1.5 if price_std / price_mean < 0.02 else 1.0))
                price_grid = np.linspace(final_prices.min(), final_prices.max(), 1000)
                u = kde(price_grid)
                u /= np.trapz(u, price_grid) + 1e-10
                with st.spinner("Computing geodesic path..."):
                    delta_p = prices[-1] - p0
                    recent_returns = returns[-min(24, len(returns)):].mean() / 100 if len(returns) > 1 else 0
                    y0 = np.concatenate([np.array([0.0, p0]), np.array([1.0, delta_p / T + recent_returns])])
                    t_eval = np.linspace(0, T, min(N, 100))
                    metric = VolatilityMetric(sigma, t, T)
                    try:
                        sol = solve_ivp(geodesic_equation, [0, T], y0, args=(metric,), t_eval=t_eval, rtol=1e-5, method='Radau')
                        geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
                    except:
                        st.warning("Geodesic computation failed. Using linear path.")
                        linear_times = np.linspace(0, T, min(N, 100))
                        linear_prices = np.linspace(p0, prices[-1], min(N, 100))
                        volatility_adjustment = np.interp(linear_times, t, sigma)
                        adjusted_prices = linear_prices * (1 + 0.1 * volatility_adjustment)
                        geodesic_df = pd.DataFrame({"Time": linear_times, "Price": adjusted_prices, "Path": "Geodesic"})
                geodesic_price = np.interp(T, geodesic_df["Time"], geodesic_df["Price"]) if not geodesic_df.empty else price_mean
                geodesic_weights = np.exp(-np.abs(price_grid - geodesic_price) / (2 * price_std))
                u_weighted = u * geodesic_weights
                from scipy.ndimage import gaussian_filter1d
                u_smooth = gaussian_filter1d(u_weighted, sigma=2)
                peak_height = np.percentile(u_smooth, 75)
                peak_distance = max(10, len(price_grid) // 50)
                peaks, _ = find_peaks(u_smooth, height=peak_height, distance=peak_distance)
                if len(peaks) < 4:
                    peaks, _ = find_peaks(u_smooth, height=0.01 * u_smooth.max(), distance=peak_distance // 2)
                levels = price_grid[peaks]
                warning_message = None
                if len(peaks) < 4:
                    warning_message = "Insufficient peaks detected. Using DBSCAN clustering."
                    X = final_prices.reshape(-1, 1)
                    db = DBSCAN(eps=price_std / 2, min_samples=50).fit(X)
                    labels = db.labels_
                    unique_labels = set(labels) - {-1}
                    if unique_labels:
                        cluster_centers = [np.mean(final_prices[labels == label]) for label in unique_labels]
                        levels = np.sort(cluster_centers)[:6]
                    else:
                        top_indices = np.argsort(u_smooth)[-4:]
                        levels = np.sort(price_grid[top_indices])
                median_of_peaks = np.median(levels)
                support_levels = levels[levels <= median_of_peaks][:2]
                resistance_levels = levels[levels > median_of_peaks][-2:]
                path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"}
                             for i in range(min(n_paths, n_display_paths)) for j in range(N)]
                plot_df = pd.DataFrame(path_data)
                if not geodesic_df.empty:
                    plot_df = pd.concat([plot_df, geodesic_df])
                if plot_df.empty or plot_df[['Time', 'Price']].isna().any().any():
                    st.error("Main chart data is empty or contains NaN values.")
                else:
                    support_df = pd.DataFrame({"Price": support_levels})
                    resistance_df = pd.DataFrame({"Price": resistance_levels})
                    base = alt.Chart(plot_df).encode(
                        x=alt.X("Time:Q", title="Time (hours)"),
                        y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False))
                    )
                    path_lines = base.mark_line(opacity=0.2).encode(
                        color=alt.value('gray'), detail='Path:N'
                    ).transform_filter(alt.datum.Path != "Geodesic")
                    geodesic_line = base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic")
                    support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
                    resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                    chart = (path_lines + geodesic_line + support_lines + resistance_lines).properties(
                        title="Price Paths, Geodesic, and S/R Grid", height=500
                    ).interactive()
                    try:
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to render main chart: {e}")
        with col2:
            st.subheader("Projected S/R Probabilities")
            epsilon = epsilon_factor * np.std(final_prices)
            if np.isnan(epsilon):
                st.error("Invalid epsilon value for probability zones.")
            else:
                support_probs = get_hit_prob(support_levels, price_grid, u, metric, T, epsilon, geodesic_df)
                resistance_probs = get_hit_prob(resistance_levels, price_grid, u, metric, T, epsilon, geodesic_df)
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

        st.header("Historical Context: Volume-by-Price Analysis")
        st.markdown("""
        High-volume price levels act as strong support or resistance.  
        Green (support) and red (resistance) zones show simulated S/R levels.  
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

        # --- Options Analysis ---
        if sel_expiry:
            st.header("Options-Based Probability Analysis")
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
                with st.spinner("Calibrating Arbitrage-Free SABR..."):
                    alpha, rho, nu, calib_error, n_points = calibrate_arbitrage_free_sabr(
                        df_options, forward_price, ttm, beta_sabr, sr_pct, use_oi_weights, r_rate
                    )
                if alpha is not None:
                    st.subheader("SABR Calibration Results")
                    sabr_cols = st.columns(4)
                    sabr_cols[0].metric("Alpha", f"{alpha:.3f}")
                    sabr_cols[1].metric("Rho", f"{rho:.3f}")
                    sabr_cols[2].metric("Nu", f"{nu:.3f}")
                    sabr_cols[3].metric("Calibration Error", f"{calib_error:.2e}")
                    strikes = np.linspace(df_options['strike'].min(), df_options['strike'].max(), 100)
                    call_prices, put_prices, pdf_df, QL, QR = arbitrage_free_sabr_pricer(
                        alpha, beta_sabr, rho, nu, forward_price, ttm, strikes, r_rate
                    )
                    df_options['model_price'] = df_options.apply(
                        lambda r: call_prices[np.argmin(np.abs(strikes - r['strike']))] if r['type'] == 'C' else
                                  put_prices[np.argmin(np.abs(strikes - r['strike']))],
                        axis=1
                    )
                    df_options['svi_dollar_mispricing'] = df_options['mark_price'] - df_options['model_price']
                    st.subheader("Implied Probability Density (Options)")
                    pdf_chart = alt.Chart(pdf_df[pdf_df['pdf'] > 1e-7]).mark_area(opacity=0.7, color='purple').encode(
                        x=alt.X('strike:Q', title='Price ($)'),
                        y=alt.Y('pdf:Q', title='Probability Density', axis=alt.Axis(format='.2e')),
                        tooltip=['strike', alt.Tooltip('pdf:Q', format='.4e')]
                    ).properties(title=f"Options Implied PDF for {sel_expiry}", width=700)
                    st.altair_chart(pdf_chart.interactive(), use_container_width=True)
                    st.subheader("Trade Ideas (Options-Based)")
                    trade_ideas = generate_trade_ideas(df_options, spot_price, top_n=3)
                    if not trade_ideas.empty:
                        trade_chart = alt.Chart(trade_ideas).mark_bar().encode(
                            y=alt.Y('display_name:N', title='Trade', sort=None),
                            x=alt.X('score:Q', title='Score'),
                            color=alt.Color('candidate_type:N', title='Strategy'),
                            tooltip=['display_name', 'details', 'rank']
                        ).properties(title="Top Options Trade Ideas", width=700)
                        st.altair_chart(trade_chart.interactive(), use_container_width=True)
                    else:
                        st.info("No viable trade ideas found based on current mispricing.")
                    st.subheader("Options Chain")
                    st.dataframe(df_options[['instrument', 'strike', 'type', 'mark_price', 'iv', 'delta', 'svi_dollar_mispricing']].style.format({
                        'strike': '{:,.0f}',
                        'mark_price': '${:.2f}',
                        'iv': '{:.2%}',
                        'delta': '{:.2f}',
                        'svi_dollar_mispricing': '${:.2f}'
                    }))
                else:
                    st.error("SABR calibration failed.")
            else:
                st.error("No valid options data available.")
        else:
            st.info("Select an options expiry to enable options analysis.")

        # --- Export Results ---
        st.header("Export Results")
        export_button = st.button("Download Charts and Data")
        if export_button:
            with st.spinner("Generating exports..."):
                try:
                    chart.save("main_chart.html")
                    with open("main_chart.html", "rb") as f:
                        st.download_button("Download Main Chart", f, file_name="main_chart.html")
                except Exception as e:
                    st.warning(f"Failed to export main chart: {e}")
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
