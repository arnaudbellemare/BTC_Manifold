import streamlit as st
import ccxt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import inv, LinAlgError
from geomstats.geometry.riemannian_metric import RiemannianMetric
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# --- Page Title and Introduction ---
st.title("Multi-Asset Market Analysis on a Fisher Information Manifold")
st.write("""
This advanced model analyzes the market as a geometric space defined by the prices of BTC and ETH.
- **Dimensionality**: The model operates in a 3D space of **(Time, BTC Price, ETH Price)**.
- **Metric Tensor**: The geometry is defined by the **Fisher Information Metric**, proxied by the inverse of the rolling 2x2 covariance matrix of asset returns (`Σ⁻¹`). This captures the assets' volatility and correlation.
- **Volume Weighting**: The metric is further scaled by total market volume. High volume (high liquidity) reduces the geometric "distance," making movement "easier."
- **Geodesic Path**: The orange line in the 3D plot represents the geodesic—the most efficient, "straightest" possible path between two points in this curved market space. It represents an idealized optimal rebalancing path.
""")

# --- Manifold Definition ---
class FisherVolumeMetric(RiemannianMetric):
    """
    A 3D Manifold (t, p1, p2) where the metric on the price sub-manifold (p1, p2)
    is the time-varying, volume-weighted inverse covariance matrix.
    """
    def __init__(self, inv_cov_series, volume_factor_series):
        super().__init__(dim=3)
        self.inv_cov_series = inv_cov_series
        self.volume_factor_series = volume_factor_series
        self.n_times = len(inv_cov_series)

    def set_time_params(self, t_max):
        self.t_max = t_max

    def get_metric_at_time_index(self, idx):
        idx = int(np.clip(idx, 0, self.n_times - 1))
        g_fisher = self.inv_cov_series.iloc[idx].values
        vol_factor = self.volume_factor_series.iloc[idx]
        g_price_space = g_fisher * vol_factor
        g = np.eye(3)
        g[1:, 1:] = g_price_space
        return g

    def metric_matrix(self, base_point):
        t = base_point[0]
        time_index = t / self.t_max * (self.n_times - 1) if self.t_max > 0 else 0
        return self.get_metric_at_time_index(time_index)

    def christoffel_symbols(self, base_point):
        eps = 1e-4
        t = base_point[0]
        
        time_index = t / self.t_max * (self.n_times - 1) if self.t_max > 0 else 0
        
        # dg/dt (numerical derivative)
        g_plus = self.get_metric_at_time_index(time_index + eps / self.t_max * (self.n_times - 1) if self.t_max > 0 else 0)
        g_minus = self.get_metric_at_time_index(time_index - eps / self.t_max * (self.n_times - 1) if self.t_max > 0 else 0)
        dg_dt = (g_plus - g_minus) / (2 * eps)

        try:
            g_inv = inv(self.metric_matrix(base_point))
        except LinAlgError:
            g_inv = np.eye(3) # Fallback to Euclidean if metric is singular

        # Γ^k_ij = 0.5 * g^kl * (∂g_li/∂x^j + ∂g_lj/∂x^i - ∂g_ij/∂x^l)
        # Simplified because g only depends on x^0 = t
        gamma = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                term = np.zeros(3)
                dg_ij_dt = dg_dt[i, j]
                if i == 0: term += dg_dt[:, j]
                if j == 0: term += dg_dt[:, i]
                term[0] -= dg_ij_dt
                gamma[:, i, j] = 0.5 * np.dot(g_inv, term)
        return gamma

def geodesic_equation(s, y, metric_obj):
    pos, vel = y[:3], y[3:]
    gamma = metric_obj.christoffel_symbols(pos)
    accel = -np.einsum('ijk,j,k->i', gamma, vel, vel)
    return np.concatenate([vel, accel])

# --- Data Fetching and Processing ---
@st.cache_data
def fetch_and_process_data(symbols=['BTC/USD', 'ETH/USD'], timeframe='1h', days=30):
    exchange = ccxt.kraken()
    end_date = pd.Timestamp.now(tz='UTC')
    start_date = end_date - pd.Timedelta(days=days)
    
    all_data = {}
    for symbol in symbols:
        try:
            since = int(start_date.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=days*24)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            all_data[symbol.split('/')[0]] = df[['close', 'volume']]
        except Exception as e:
            st.error(f"Could not fetch data for {symbol}: {e}")
            return None
    
    if len(all_data) != len(symbols): return None
        
    df_merged = pd.concat(all_data, axis=1)
    df_merged.columns = ['_'.join(col).strip() for col in df_merged.columns.values]
    df_merged.dropna(inplace=True)
    return df_merged

# --- NEW: S/R Analysis from Path Distribution ---
def analyze_path_distribution(final_prices, asset_name):
    """
    Analyzes the distribution of final prices from a Monte Carlo simulation
    to find support and resistance levels.
    This replaces the 1D Fokker-Planck solver with a more robust method.
    """
    st.subheader(f"Support & Resistance Analysis for {asset_name}")

    # 1. Estimate probability density using Kernel Density Estimation (KDE)
    kde = gaussian_kde(final_prices)
    price_grid = np.linspace(final_prices.min(), final_prices.max(), 400)
    u = kde(price_grid)
    dp = price_grid[1] - price_grid[0]

    # 2. Find S/R levels by analyzing derivatives of the density curve
    du = np.gradient(u, dp)
    d2u = np.gradient(du, dp)

    # Inflection points where concavity changes are potential S/R levels
    support_idx, _ = find_peaks(-d2u, height=np.std(d2u)*0.5, distance=5)
    resistance_idx, _ = find_peaks(d2u, height=np.std(d2u)*0.5, distance=5)

    support_levels = price_grid[support_idx]
    resistance_levels = price_grid[resistance_idx]
    
    # Fallback if insufficient levels are found
    if len(support_levels) < 1 or len(resistance_levels) < 1:
        st.warning("Derivative method found few levels. Using density peaks as fallback.")
        peaks, _ = find_peaks(u, height=0.1 * u.max(), distance=10)
        levels = price_grid[peaks]
        median_level = np.median(levels) if len(levels) > 0 else np.median(price_grid)
        support_levels = levels[levels <= median_level]
        resistance_levels = levels[levels > median_level]

    # 3. Display Results
    density_df = pd.DataFrame({'Price': price_grid, 'Density': u})
    s_df = pd.DataFrame({'Level': support_levels})
    r_df = pd.DataFrame({'Level': resistance_levels})

    base = alt.Chart(density_df).mark_area(opacity=0.3).encode(
        x=alt.X('Price:Q', title=f'{asset_name} Price'),
        y='Density:Q'
    )
    s_lines = alt.Chart(s_df).mark_rule(color='green', strokeWidth=2).encode(y='Level:Q')
    r_lines = alt.Chart(r_df).mark_rule(color='red', strokeWidth=2).encode(y='Level:Q')

    chart = (base + s_lines + r_lines).properties(
        title=f"Final Price Probability Density with Support (Green) and Resistance (Red)"
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.write("**Support Levels:**")
    st.dataframe(s_df.style.format({'Level': '${:,.2f}'}))
    st.write("**Resistance Levels:**")
    st.dataframe(r_df.style.format({'Level': '${:,.2f}'}))


# --- Main App Logic ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 10, 90, 30)
rolling_window = st.sidebar.slider("Rolling Window for Covariance (Hours)", 12, 168, 72)
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 10000, 2000, step=100)
projection_hours = st.sidebar.slider("Projection Horizon (Hours)", 12, 168, 72)

# Load and process data
df = fetch_and_process_data(days=days_history)

if df is not None:
    # Calculate returns and covariance
    returns = df[['BTC_close', 'ETH_close']].pct_change().dropna()
    cov_series = returns.rolling(window=rolling_window).cov().unstack()
    cov_series.columns = ['BTC_BTC', 'BTC_ETH', 'ETH_BTC', 'ETH_ETH']
    cov_series = cov_series[['BTC_BTC', 'BTC_ETH', 'ETH_ETH']].dropna()

    # Calculate inverse covariance (Fisher Metric)
    inv_cov_list = []
    for idx, row in cov_series.iterrows():
        cov_matrix = np.array([[row.BTC_BTC, row.BTC_ETH], [row.ETH_BTC, row.ETH_ETH]])
        try:
            inv_cov_list.append(pd.Series(inv(cov_matrix).flatten(), index=['inv_11', 'inv_12', 'inv_21', 'inv_22'], name=idx))
        except LinAlgError:
            inv_cov_list.append(pd.Series(np.eye(2).flatten(), index=['inv_11', 'inv_12', 'inv_21', 'inv_22'], name=idx))
    inv_cov_series = pd.DataFrame(inv_cov_list)

    # Calculate volume factor
    total_volume = df['BTC_volume'] * df['BTC_close'] + df['ETH_volume'] * df['ETH_close']
    volume_factor = 1 / (1 + total_volume / total_volume.mean())
    volume_factor = volume_factor.rolling(window=rolling_window).mean().dropna()

    # Align all data series
    common_index = cov_series.index.intersection(inv_cov_series.index).intersection(volume_factor.index)
    inv_cov_series = inv_cov_series.loc[common_index]
    volume_factor = volume_factor.loc[common_index]
    aligned_prices = df.loc[common_index, ['BTC_close', 'ETH_close']]

    # Initialize Metric and Geodesic
    metric = FisherVolumeMetric(inv_cov_series, volume_factor)
    T_total_hours = (aligned_prices.index[-1] - aligned_prices.index[0]).total_seconds() / 3600
    metric.set_time_params(T_total_hours)
    
    initial_point = np.array([0.0, aligned_prices['BTC_close'].iloc[0], aligned_prices['ETH_close'].iloc[0]])
    final_point = np.array([T_total_hours, aligned_prices['BTC_close'].iloc[-1], aligned_prices['ETH_close'].iloc[-1]])
    initial_velocity = (final_point - initial_point) / T_total_hours if T_total_hours > 0 else np.zeros(3)
    y0 = np.concatenate([initial_point, initial_velocity])

    with st.spinner("Calculating geodesic path on the manifold..."):
        t_eval = np.linspace(0, T_total_hours, len(aligned_prices))
        sol = solve_ivp(geodesic_equation, [0, T_total_hours], y0, args=(metric,), t_eval=t_eval, rtol=1e-4, atol=1e-6, method='RK45')
        geodesic_df = pd.DataFrame(sol.y.T, columns=['time', 'BTC', 'ETH', 'v_t', 'v_btc', 'v_eth'])

    # Monte Carlo Simulation
    with st.spinner("Running Multi-Asset Monte Carlo Simulation..."):
        dt = projection_hours / len(aligned_prices)
        mu = returns.mean().values
        last_cov = returns.iloc[-rolling_window:].cov().values
        L = np.linalg.cholesky(last_cov) # Cholesky decomposition for correlated noise

        p0 = aligned_prices.iloc[-1].values
        num_steps = int(projection_hours / (T_total_hours / len(aligned_prices)))
        paths = np.zeros((n_paths, num_steps, 2))
        paths[:, 0, :] = p0

        for i in range(1, num_steps):
            Z = np.random.normal(size=(n_paths, 2))
            correlated_dW = Z @ L.T
            paths[:, i, :] = paths[:, i-1, :] * (1 + mu * dt + np.sqrt(dt) * correlated_dW)

    # 3D Visualization
    fig = go.Figure()
    # Historical Path
    fig.add_trace(go.Scatter3d(
        x=aligned_prices.index, y=aligned_prices['BTC_close'], z=aligned_prices['ETH_close'],
        mode='lines', line=dict(color='blue', width=4), name='Historical Path'
    ))
    # Geodesic Path
    fig.add_trace(go.Scatter3d(
        x=aligned_prices.index, y=geodesic_df['BTC'], z=geodesic_df['ETH'],
        mode='lines', line=dict(color='orange', width=6), name='Geodesic (Optimal) Path'
    ))
    
    # Projected Paths
    proj_times = pd.to_datetime(aligned_prices.index[-1]) + pd.to_timedelta(np.arange(num_steps) * dt, 'h')
    for i in range(min(n_paths, 100)): # Display a subset of paths
        fig.add_trace(go.Scatter3d(
            x=proj_times, y=paths[i, :, 0], z=paths[i, :, 1],
            mode='lines', line=dict(color='rgba(128,128,128,0.3)'), showlegend=False
        ))

    fig.update_layout(
        title='Market Manifold: Historical, Geodesic, and Projected Paths',
        scene=dict(xaxis_title='Time', yaxis_title='BTC Price (USD)', zaxis_title='ETH Price (USD)'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display S/R Analysis using the new robust method
    col1, col2 = st.columns(2)
    with col1:
        analyze_path_distribution(paths[:, -1, 0], 'BTC')
    with col2:
        analyze_path_distribution(paths[:, -1, 1], 'ETH')

else:
    st.error("Failed to load data. Please check connection or try again later.")
