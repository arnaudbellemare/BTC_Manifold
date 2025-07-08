import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.euclidean
import geomstats.geometry.poincare_ball
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import warnings
import time
from datetime import datetime, timedelta

# --- Global Settings ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("BTC/USD Price Analysis on a Volatility-Weighted Manifold")
st.write("""
This application models the Bitcoin market as a 2D geometric space (manifold) of (Time, Price), 
warped by market volatility estimated via a GARCH model. 
- **High Volatility (Yellow areas):** Stretched regions indicate high risk and large price changes.
- **Low Volatility (Dark areas):** Flatter regions represent calmer market periods.
- **Geodesic (Red Line):** The shortest path through the manifold, optimized using the learning-rate-free RDoG algorithm.
- **S/R Grid:** Green (support) and red (resistance) lines show probable price levels from Monte Carlo simulations.
""")

# --- Configuration ---
CONFIG = {
    "SCALING_FACTOR": 10000,
    "DEFAULT_DAYS_HISTORY": 30,
    "MIN_DAYS_HISTORY": 7,
    "MAX_DAYS_HISTORY": 90,
    "MIN_PATHS": 500,
    "MAX_PATHS": 10000,
    "DEFAULT_PATHS": 2000,
    "MIN_DISPLAY_PATHS": 10,
    "MAX_DISPLAY_PATHS": 200,
    "DEFAULT_DISPLAY_PATHS": 50,
    "MIN_LEVELS": 2,
    "MAX_LEVELS": 10,
    "DEFAULT_LEVELS": 4,
    "MIN_EPSILON_FACTOR": 0.1,
    "MAX_EPSILON_FACTOR": 1.0,
    "DEFAULT_EPSILON_FACTOR": 0.25,
    "RDoG_STEPS": 100,
    "RDoG_EPSILON": 1e-3,
    "API_RETRIES": 3
}

# --- Geometric Modeling Class ---
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T, manifold_type="Euclidean"):
        self.dim = 2
        if manifold_type == "Poincaré":
            self.space = geomstats.geometry.poincare_ball.PoincareBall(dim=self.dim)
        else:
            self.space = geomstats.geometry.euclidean.Euclidean(dim=self.dim)
        super().__init__(space=self.space)
        self.sigma = sigma
        self.t = t
        self.T = T
        self.manifold_type = manifold_type

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6)
        if self.manifold_type == "Poincaré":
            norm = np.linalg.norm(base_point)
            if norm >= 1:
                norm = 0.9999  # Prevent singularity
            conformal_factor = 4 / (1 - norm**2)**2
            return conformal_factor * np.diag([1.0, sigma_val**2])
        return np.diag([1.0, sigma_val**2])

    def christoffel_symbols(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6)
        eps = 1e-6
        t_plus = min(t_val + eps, self.T)
        t_minus = max(t_val - eps, 0)
        idx_plus = int(np.clip(t_plus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        idx_minus = int(np.clip(t_minus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        d_sigma_dt = (self.sigma[idx_plus] - self.sigma[idx_minus]) / (2 * eps)
        gamma = np.zeros((2, 2, 2))
        gamma[1, 0, 1] = (1 / sigma_val) * d_sigma_dt
        gamma[1, 1, 0] = gamma[1, 0, 1]
        gamma[0, 1, 1] = -sigma_val * d_sigma_dt
        if self.manifold_type == "Poincaré":
            norm = np.linalg.norm(base_point)
            if norm >= 1:
                norm = 0.9999
            conformal_factor = 4 / (1 - norm**2)**2
            gamma *= conformal_factor
        return gamma

# --- RDoG Optimization for Geodesic ---
def rdog_geodesic(metric, x0, xT, T, n_steps=CONFIG["RDoG_STEPS"], epsilon=CONFIG["RDoG_EPSILON"]):
    x = np.array([0.0, x0[1]])  # Start at (t=0, p0)
    v = np.array([T, xT[1] - x0[1]]) / T if T > 0 else np.array([0.0, 0.0])  # Initial velocity
    Gt = 0.0
    rt_bar = epsilon
    path = [x.copy()]
    
    # Estimate curvature (simplified for demo; can be computed from volatility data)
    kappa = -0.01  # Negative curvature for volatility-weighted manifold
    def zeta_kappa(d):
        if kappa < 0:
            return np.sqrt(abs(kappa)) * d / np.tanh(np.sqrt(abs(kappa)) * d)
        return 1.0

    for t in range(n_steps):
        g = np.einsum('ijk,j,k->i', metric.christoffel_symbols(x), v, v)  # Gradient from geodesic equation
        norm_g = np.sqrt(metric.inner_product(g, g, x))
        Gt += norm_g**2
        rt = np.sqrt(np.sum((x - x0)**2))  # Proxy for geodesic distance
        rt_bar = max(epsilon, rt)
        eta_t = rt_bar / np.sqrt(zeta_kappa(rt_bar) * max(Gt, 1e-6))
        x = metric.exp(x, -eta_t * g)
        v = metric.parallel_transport(v, x, -eta_t * g)
        path.append(x.copy())
    
    return pd.DataFrame({
        "Time": [p[0] for p in path],
        "Price": [p[1] for p in path],
        "Path": "Geodesic"
    })

# --- Helper Functions ---
@st.cache_data
def fetch_kraken_data(symbol, timeframe, start_date, end_date, retries=CONFIG["API_RETRIES"]):
    exchange = ccxt.kraken()
    since = int(start_date.timestamp() * 1000)
    limit = int((end_date - start_date).total_seconds() / 3600) + 24
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
            if len(df) < 10:
                st.warning(f"Insufficient data points ({len(df)}) for {symbol}.")
                return None
            if df['close'].isna().any() or (df['close'] <= 0).any():
                st.warning("Invalid data detected (NaN or negative prices).")
                return None
            st.success(f"Fetched {len(df)} data points for {symbol}.")
            return df
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                st.error("Failed to fetch data after retries. Using simulated data.")
                sim_t = pd.date_range(start=start_date, end=end_date, freq='h')
                sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, len(sim_t)))
                return pd.DataFrame({'datetime': sim_t, 'close': sim_prices})
    return None

@st.cache_data
def simulate_paths(p0, mu, sigma, T, N, n_paths):
    if N < 2:
        return np.array([[p0]]), np.array([0])
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))
    paths[:, 0] = p0
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N - 1))
    log_returns = (mu - 0.5 * sigma[:-1]**2) * dt + sigma[:-1] * dW
    paths[:, 1:] = p0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths, t

def compute_sr_levels(final_prices, n_levels=CONFIG["DEFAULT_LEVELS"]):
    kde = gaussian_kde(final_prices)
    price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
    u = kde(price_grid)
    u /= np.trapz(u, price_grid)
    peaks, _ = find_peaks(u, height=0.05 * u.max(), distance=len(price_grid) // 25)
    levels = price_grid[peaks]
    
    if len(levels) >= 2:
        median_of_peaks = np.median(levels)
        support_levels = levels[levels <= median_of_peaks]
        resistance_levels = levels[levels > median_of_peaks]
    else:
        st.warning("Few distinct peaks found. Using K-means clustering for S/R levels.")
        kmeans = KMeans(n_clusters=n_levels, random_state=0).fit(final_prices.reshape(-1, 1))
        levels = np.sort(kmeans.cluster_centers_.flatten())
        median = np.median(levels)
        support_levels = levels[levels <= median]
        resistance_levels = levels[levels > median]
    
    if len(resistance_levels) == 0 and len(support_levels) > 1:
        resistance_levels = np.array([support_levels[-1]])
        support_levels = support_levels[:-1]
    if len(support_levels) == 0 and len(resistance_levels) > 1:
        support_levels = np.array([resistance_levels[0]])
        resistance_levels = resistance_levels[1:]
    
    return support_levels, resistance_levels

def visualize_manifold(metric, t_grid, p_grid):
    st.subheader("Market Manifold Geometry")
    st.write("Heatmap shows the 'cost' of price movement (proportional to σ²). Yellow = high-volatility, stretched regions. Dark = low-volatility, flat regions.")
    g_pp_values = []
    for t_val in t_grid:
        cost = metric.metric_matrix([t_val, 0])[1, 1]
        scaled_cost = cost * CONFIG["SCALING_FACTOR"]
        for p_val in p_grid:
            g_pp_values.append({'Time': t_val, 'Price': p_val, 'Cost': scaled_cost})
    g_df = pd.DataFrame(g_pp_values)
    heatmap = alt.Chart(g_df).mark_rect().encode(
        x=alt.X('Time:Q', title='Time (hours)'),
        y=alt.Y('Price:Q', title='BTC/USD Price', scale=alt.Scale(zero=False)),
        color=alt.Color('Cost:Q', scale=alt.Scale(scheme='viridis'),
                        legend=alt.Legend(title=f"Cost (σ² × {CONFIG['SCALING_FACTOR']})")),
        tooltip=[alt.Tooltip('Time:Q', title='Time (hours)', format='.2f'),
                 alt.Tooltip('Price:Q', title='Price ($)', format='.2f'),
                 alt.Tooltip('Cost:Q', title='Cost', format='.2f')]
    ).properties(title="Volatility Landscape", height=300).interactive()
    return heatmap

# --- Main Application Logic ---
class ManifoldAnalyzer:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.metric = None
        self.paths = None
        self.t = None
        self.support_levels = None
        self.resistance_levels = None
        self.support_probs = None
        self.resistance_probs = None
        self.geodesic_df = None

    def fetch_and_process_data(self, symbol, timeframe, start_date, end_date):
        with st.spinner("Fetching market data..."):
            self.df = fetch_kraken_data(symbol, timeframe, start_date, end_date)
            if self.df is None or len(self.df) < 10:
                st.error("Could not load or process data.")
                return False
        
        prices = self.df['close'].values
        times_pd = self.df['datetime']
        self.t = (times_pd - times_pd.iloc[0]).dt.total_seconds() / 3600
        self.T = self.t.iloc[-1]
        self.N = len(prices)
        self.p0 = prices[0]
        returns = 100 * self.df['close'].pct_change().dropna()
        
        with st.spinner("Fitting GARCH model..."):
            try:
                model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
                self.sigma = model.conditional_volatility / 100
                self.sigma = np.pad(self.sigma, (self.N - len(self.sigma), 0), mode='edge')
            except Exception:
                st.warning("GARCH model failed. Using constant volatility.")
                self.sigma = np.full(self.N, returns.std() / 100)
        self.mu = returns.mean() / 100
        return True

    def simulate_and_analyze(self, n_paths, n_levels, epsilon_factor):
        with st.spinner("Simulating price paths..."):
            self.paths, self.t = simulate_paths(self.p0, self.mu, self.sigma, self.T, self.N, n_paths)
        
        final_prices = self.paths[:, -1]
        self.support_levels, self.resistance_levels = compute_sr_levels(final_prices, n_levels)
        final_std_dev = np.std(final_prices)
        epsilon = epsilon_factor * final_std_dev
        
        self.metric = VolatilityMetric(self.sigma, self.t, self.T, manifold_type=st.session_state.manifold_type)
        
        def get_hit_prob(level_list):
            kde = gaussian_kde(final_prices)
            price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
            u = kde(price_grid)
            u /= np.trapz(u, price_grid)
            probs = []
            for level in level_list:
                mask = (price_grid >= level - epsilon) & (price_grid <= level + epsilon)
                raw_prob = np.trapz(u[mask], price_grid[mask])
                volume_element = np.sqrt(np.abs(np.linalg.det(self.metric.metric_matrix([self.T, level]))))
                probs.append(raw_prob * volume_element)
            total_prob = sum(probs)
            return [p / total_prob for p in probs] if total_prob > 0 else [0] * len(probs)
        
        self.support_probs = get_hit_prob(self.support_levels)
        self.resistance_probs = get_hit_prob(self.resistance_levels)
        
        with st.spinner("Computing geodesic path with RDoG..."):
            self.geodesic_df = rdog_geodesic(self.metric, [0.0, self.p0], [self.T, self.df['close'].iloc[-1]], self.T)

    def visualize(self, n_display_paths):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            path_data = [{"Time": self.t[j], "Price": self.paths[i, j], "Path": f"Path_{i}"}
                         for i in range(min(self.paths.shape[0], n_display_paths))
                         for j in range(self.N)]
            plot_df = pd.concat([pd.DataFrame(path_data), self.geodesic_df])
            support_df = pd.DataFrame({"Price": self.support_levels})
            resistance_df = pd.DataFrame({"Price": self.resistance_levels})
            
            base = alt.Chart(plot_df).encode(
                x=alt.X("Time:Q", title="Time (hours)"),
                y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip('Time:Q', title='Time (hours)', format='.2f'),
                         alt.Tooltip('Price:Q', title='Price ($)', format='.2f')]
            )
            path_lines = base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic")
            geodesic_line = base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic")
            support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
            resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
            history_df = pd.DataFrame({'Time': self.t, 'Price': self.df['close']})
            history_line = alt.Chart(history_df).mark_line(color='white', strokeWidth=2.5, opacity=0.7).encode(x='Time:Q', y='Price:Q')
            main_chart = (path_lines + geodesic_line + support_lines + resistance_lines + history_line).properties(
                title="Price Paths, Geodesic, and S/R Grid", height=500
            ).interactive()
            st.altair_chart(main_chart, use_container_width=True)
        
        with col2:
            viz_p_grid = np.linspace(self.df['close'].min(), self.df['close'].max(), 50)
            manifold_heatmap = visualize_manifold(self.metric, self.t, viz_p_grid)
            st.altair_chart((manifold_heatmap + history_line).properties(height=300).interactive(), use_container_width=True)
            
            st.subheader("Analysis Summary")
            st.metric("Expected Final Price", f"${np.mean(self.paths[:, -1]):,.2f}")
            st.write("**Support Levels (BTC/USD):**")
            if self.support_levels.size > 0:
                st.dataframe(pd.DataFrame({
                    'Level': self.support_levels,
                    'Hit Probability': self.support_probs
                }).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
            else:
                st.write("No distinct support levels found.")
            st.write("**Resistance Levels (BTC/USD):**")
            if self.resistance_levels.size > 0:
                st.dataframe(pd.DataFrame({
                    'Level': self.resistance_levels,
                    'Hit Probability': self.resistance_probs
                }).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
            else:
                st.write("No distinct resistance levels found.")

# --- Sidebar Controls ---
st.sidebar.header("Model Parameters")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=CONFIG["DEFAULT_DAYS_HISTORY"]))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
n_paths = st.sidebar.slider("Number of Simulated Paths", CONFIG["MIN_PATHS"], CONFIG["MAX_PATHS"], CONFIG["DEFAULT_PATHS"], step=100)
n_display_paths = st.sidebar.slider("Number of Paths to Display", CONFIG["MIN_DISPLAY_PATHS"], CONFIG["MAX_DISPLAY_PATHS"], CONFIG["DEFAULT_DISPLAY_PATHS"], step=10)
n_levels = st.sidebar.slider("Number of S/R Levels", CONFIG["MIN_LEVELS"], CONFIG["MAX_LEVELS"], CONFIG["DEFAULT_LEVELS"])
epsilon_factor = st.sidebar.slider("Probability Range Factor (for Hit %)", CONFIG["MIN_EPSILON_FACTOR"], CONFIG["MAX_EPSILON_FACTOR"], CONFIG["DEFAULT_EPSILON_FACTOR"], step=0.05)
manifold_type = st.sidebar.selectbox("Manifold Type", ["Euclidean", "Poincaré"], key="manifold_type")

# --- Run Analysis ---
analyzer = ManifoldAnalyzer(CONFIG)
if st.sidebar.button("Run Analysis"):
    start_date = pd.Timestamp(start_date, tz='UTC')
    end_date = pd.Timestamp(end_date, tz='UTC')
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        if analyzer.fetch_and_process_data('BTC/USD', '1h', start_date, end_date):
            analyzer.simulate_and_analyze(n_paths, n_levels, epsilon_factor)
            analyzer.visualize(n_display_paths)
