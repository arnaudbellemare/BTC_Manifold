import streamlit as st
import ccxt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import inv, LinAlgError
# Import the specific classes we need
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import Euclidean
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings
import geomstats

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# Debug: Check geomstats version (This is good to keep)
st.write("geomstats version:", geomstats.__version__)

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
        # --- START OF THE FIX ---
        # 1. Define the dimension of the manifold.
        self.dim = 3
        
        # 2. Create an instance of the underlying space (a 3D Euclidean space).
        #    This is the required 'space' argument for the parent class.
        space = Euclidean(dim=self.dim)
        
        # 3. Call the parent class's __init__ with the mandatory 'space' argument.
        super().__init__(space=space)
        # --- END OF THE FIX ---
        
        self.inv_cov_series = inv_cov_series
        self.volume_factor_series = volume_factor_series
        self.n_times = len(inv_cov_series)

        # Validate inputs
        if self.inv_cov_series.empty or self.volume_factor_series.empty:
            raise ValueError("inv_cov_series or volume_factor_series is empty")
        if len(self.inv_cov_series) != len(self.volume_factor_series):
            raise ValueError("inv_cov_series and volume_factor_series have mismatched lengths")

    def set_time_params(self, t_max):
        """Set the time scale for normalization."""
        self.t_max = t_max

    def get_metric_at_time_index(self, idx):
        """Helper to get the metric tensor at a specific time index."""
        # Clamp index to be within the bounds of the series
        idx = int(np.clip(idx, 0, self.n_times - 1))
        
        # Retrieve pre-computed inverse covariance and volume factor
        g_fisher = self.inv_cov_series.iloc[idx].values.reshape(2, 2)
        vol_factor = self.volume_factor_series.iloc[idx]
        
        # Scale the price-space part of the metric
        g_price_space = g_fisher * vol_factor
        
        # Embed the 2x2 price metric into the full 3x3 (t, p1, p2) metric
        g = np.eye(3)
        g[1:, 1:] = g_price_space
        return g

    def metric_matrix(self, base_point=None):
        """
        Computes the metric matrix at a given base point (t, p1, p2).
        The metric only depends on the time coordinate 't'.
        """
        t = base_point[0]
        # Normalize time 't' to get an index into our data series
        time_index = t / self.t_max * (self.n_times - 1) if self.t_max > 0 else 0
        return self.get_metric_at_time_index(time_index)

    def christoffel_symbols(self, base_point=None):
        """
        Computes the Christoffel symbols at a given base point.
        Uses a numerical derivative for the time-dependent part of the metric.
        """
        eps = 1e-4  # Small step for numerical differentiation
        t = base_point[0]
        
        # Normalize time 't' to get an index
        time_index = t / self.t_max * (self.n_times - 1) if self.t_max > 0 else 0
        
        # Numerical derivative of the metric tensor w.r.t. time (x^0)
        # We need a small delta in terms of our time index
        idx_delta = eps / self.t_max * (self.n_times - 1) if self.t_max > 0 else 0
        g_plus = self.get_metric_at_time_index(time_index + idx_delta)
        g_minus = self.get_metric_at_time_index(time_index - idx_delta)
        dg_dt = (g_plus - g_minus) / (2 * eps)

        try:
            g_inv = inv(self.metric_matrix(base_point))
        except LinAlgError:
            g_inv = np.eye(self.dim) # Fallback to Euclidean if metric is singular

        # Christoffel symbol formula: Γ^k_ij = 0.5 * g^kl * (∂g_li/∂x^j + ∂g_lj/∂x^i - ∂g_ij/∂x^l)
        # This simplifies greatly because our metric g only depends on x^0 = t.
        # So, derivatives w.r.t. x^1 (p1) and x^2 (p2) are zero.
        gamma = np.zeros((self.dim, self.dim, self.dim))
        
        # The only non-zero derivatives are ∂g_ij / ∂t
        for i in range(self.dim):
            for j in range(self.dim):
                # The term in parentheses simplifies to:
                # δ_j0 * ∂g_li/∂t + δ_i0 * ∂g_lj/∂t - δ_l0 * ∂g_ij/∂t
                # where δ is the Kronecker delta
                term = np.zeros(self.dim)
                dg_ij_dt = dg_dt[i, j]
                if i == 0:  # If the i-th coordinate is time
                    term += dg_dt[:, j]
                if j == 0:  # If the j-th coordinate is time
                    term += dg_dt[:, i]
                
                # This corresponds to the -∂g_ij/∂x^l term where l=0 (time)
                term[0] -= dg_ij_dt
                
                # Final calculation for Γ^k_ij
                gamma[:, i, j] = 0.5 * np.dot(g_inv, term)
                
        return gamma

def geodesic_equation(s, y, metric_obj):
    """The geodesic ODE: d²x/ds² + Γ^i_jk * (dx/ds)^j * (dx/ds)^k = 0"""
    pos, vel = y[:3], y[3:]
    gamma = metric_obj.christoffel_symbols(pos)
    # Einstein summation for the acceleration term
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
            # Fetch one more day than needed to ensure enough data for rolling window
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=(days+5)*24)
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

# --- S/R Analysis from Path Distribution ---
def analyze_path_distribution(final_prices, asset_name):
    """
    Analyzes the distribution of final prices from a Monte Carlo simulation
    to find support and resistance levels.
    """
    st.subheader(f"Support & Resistance Analysis for {asset_name}")

    if len(final_prices) < 2:
        st.warning(f"Not enough data points to analyze S/R for {asset_name}.")
        return [], []

    kde = gaussian_kde(final_prices)
    price_grid = np.linspace(final_prices.min(), final_prices.max(), 400)
    density = kde(price_grid)
    
    # Find peaks in the density function as potential S/R levels
    peaks, _ = find_peaks(density, height=0.1 * density.max(), distance=10)
    
    if len(peaks) < 2:
        st.warning("Could not find distinct S/R levels from distribution. Using quantiles as fallback.")
        support_levels = [np.quantile(final_prices, 0.25)]
        resistance_levels = [np.quantile(final_prices, 0.75)]
    else:
        levels = price_grid[peaks]
        median_level = np.median(levels)
        support_levels = levels[levels <= median_level]
        resistance_levels = levels[levels > median_level]

    density_df = pd.DataFrame({'Price': price_grid, 'Density': density})
    s_df = pd.DataFrame({'Level': support_levels})
    r_df = pd.DataFrame({'Level': resistance_levels})

    try:
        import altair as alt
        base = alt.Chart(density_df).mark_area(opacity=0.3).encode(
            x=alt.X('Price:Q', title=f'{asset_name} Price'),
            y='Density:Q'
        )
        s_lines = alt.Chart(s_df).mark_rule(color='green', strokeWidth=2).encode(x='Level:Q')
        r_lines = alt.Chart(r_df).mark_rule(color='red', strokeWidth=2).encode(x='Level:Q')

        chart = (base + s_lines + r_lines).properties(
            title=f"Final Price Probability Density with Support (Green) and Resistance (Red)"
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        st.warning("Altair not installed. Skipping density plot.")

    st.write("**Support Levels:**")
    st.dataframe(s_df.style.format({'Level': '${:,.2f}'}))
    st.write("**Resistance Levels:**")
    st.dataframe(r_df.style.format({'Level': '${:,.2f}'}))

    return support_levels, resistance_levels

# --- Main App Logic ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 10, 90, 30)
rolling_window = st.sidebar.slider("Rolling Window for Covariance (Hours)", 12, 168, 72)
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 10000, 2000, step=100)
projection_hours = st.sidebar.slider("Projection Horizon (Hours)", 12, 168, 72)

data_load_state = st.text("Loading data...")
df = fetch_and_process_data(days=days_history)
data_load_state.text("Data loaded successfully!")


if df is not None:
    # st.write("Debug: Raw data columns:", list(df.columns))
    
    # Calculate returns
    returns = df[['BTC_close', 'ETH_close']].pct_change().dropna()
    
    # Compute rolling covariance and inverse covariance
    cov_matrices = returns.rolling(window=rolling_window).cov().unstack()
    cov_matrices.dropna(inplace=True)
    
    if cov_matrices.empty:
        st.error("Could not compute covariance matrix. Try a smaller rolling window or more historical data.")
        st.stop()
        
    inv_cov_list = []
    for idx, row in cov_matrices.iterrows():
        cov_matrix = np.array([[row[('BTC_close', 'BTC_close')], row[('BTC_close', 'ETH_close')]],
                               [row[('ETH_close', 'BTC_close')], row[('ETH_close', 'ETH_close')]]])
        try:
            inv_cov = inv(cov_matrix)
            inv_cov_list.append(pd.Series(inv_cov.flatten(), name=idx))
        except LinAlgError:
            inv_cov_list.append(pd.Series(np.eye(2).flatten(), name=idx)) # Fallback
            
    inv_cov_series = pd.DataFrame(inv_cov_list)
    inv_cov_series.columns = ['inv_11', 'inv_12', 'inv_21', 'inv_22']
    
    # Calculate volume factor
    total_volume_usd = df['BTC_volume'] * df['BTC_close'] + df['ETH_volume'] * df['ETH_close']
    volume_factor = 1 / (1 + total_volume_usd / total_volume_usd.mean())
    volume_factor = volume_factor.rolling(window=rolling_window).mean().dropna()

    # Align all data series to a common time index
    common_index = inv_cov_series.index.intersection(volume_factor.index)
    inv_cov_series = inv_cov_series.loc[common_index]
    volume_factor = volume_factor.loc[common_index]
    aligned_prices = df.loc[common_index, ['BTC_close', 'ETH_close']]

    if aligned_prices.empty or inv_cov_series.empty or volume_factor.empty:
        st.error("Data alignment resulted in empty series. Please check rolling window and data period.")
        st.stop()
        
    # --- Geodesic Calculation ---
    try:
        metric = FisherVolumeMetric(inv_cov_series, volume_factor)
        
        T_total_hours = (aligned_prices.index[-1] - aligned_prices.index[0]).total_seconds() / 3600
        metric.set_time_params(T_total_hours)
        
        initial_point = np.array([0.0, aligned_prices['BTC_close'].iloc[0], aligned_prices['ETH_close'].iloc[0]])
        final_point = np.array([T_total_hours, aligned_prices['BTC_close'].iloc[-1], aligned_prices['ETH_close'].iloc[-1]])
        
        # Simple initial velocity guess
        initial_velocity = (final_point - initial_point) / T_total_hours if T_total_hours > 0 else np.zeros(3)
        y0 = np.concatenate([initial_point, initial_velocity])

        with st.spinner("Calculating geodesic path on the manifold..."):
            t_eval = np.linspace(0, T_total_hours, len(aligned_prices))
            sol = solve_ivp(
                fun=geodesic_equation, 
                t_span=[0, T_total_hours], 
                y0=y0, 
                args=(metric,), 
                t_eval=t_eval, 
                method='RK45', # RK45 is a good general-purpose solver
                rtol=1e-4, 
                atol=1e-6
            )
            geodesic_df = pd.DataFrame(sol.y.T, columns=['time', 'BTC', 'ETH', 'v_t', 'v_btc', 'v_eth'])

    except Exception as e:
        st.error(f"An error occurred during geodesic calculation: {e}")
        st.exception(e) # Print full traceback for debugging
        st.stop()

    # --- Monte Carlo Simulation ---
    with st.spinner("Running Multi-Asset Monte Carlo Simulation..."):
        # Use recent market conditions for projection
        recent_returns = returns.iloc[-rolling_window:]
        mu = recent_returns.mean().values
        last_cov = recent_returns.cov().values
        L = np.linalg.cholesky(last_cov)  # Cholesky decomposition for correlated noise

        p0 = aligned_prices.iloc[-1].values
        # Timestep should correspond to the data frequency (e.g., 1 hour)
        dt = 1 # Projecting in 1-hour steps
        num_steps = projection_hours
        
        paths = np.zeros((n_paths, num_steps + 1, 2))
        paths[:, 0, :] = p0

        for i in range(1, num_steps + 1):
            # Correlated random shocks
            Z = np.random.normal(size=(n_paths, 2))
            correlated_dW = Z @ L.T
            # Geometric Brownian Motion step
            paths[:, i, :] = paths[:, i-1, :] * np.exp((mu - 0.5 * np.diag(last_cov)) * dt + np.sqrt(dt) * correlated_dW)


    # --- Visualization Section ---
    st.header("3D Market Visualization")
    fig = go.Figure()
    # Historical Path
    fig.add_trace(go.Scatter3d(
        x=aligned_prices.index, y=aligned_prices['BTC_close'], z=aligned_prices['ETH_close'],
        mode='lines', line=dict(color='blue', width=4), name='Historical Path'
    ))
    # Geodesic Path
    fig.add_trace(go.Scatter3d(
        x=aligned_prices.index, y=geodesic_df['BTC'], z=geodesic_df['ETH'],
        mode='lines', line=dict(color='orange', width=6, dash='dash'), name='Geodesic (Optimal) Path'
    ))
    
    # Projected Paths
    proj_times = pd.date_range(start=aligned_prices.index[-1], periods=num_steps + 1, freq='h')
    for i in range(min(n_paths, 100)):  # Display a subset of paths for performance
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

    # Display S/R Analysis and collect levels
    col1, col2 = st.columns(2)
    with col1:
        btc_support_levels, btc_resistance_levels = analyze_path_distribution(paths[:, -1, 0], 'BTC')
    with col2:
        eth_support_levels, eth_resistance_levels = analyze_path_distribution(paths[:, -1, 1], 'ETH')

    # --- Price Graph with Support/Resistance Grid ---
    st.header("Price Charts with Support and Resistance Grid")
    col1, col2 = st.columns(2)
    
    full_history_btc = df.loc[aligned_prices.index[0]:, 'BTC_close']
    full_history_eth = df.loc[aligned_prices.index[0]:, 'ETH_close']

    with col1:
        fig_btc = go.Figure()
        fig_btc.add_trace(go.Scatter(x=full_history_btc.index, y=full_history_btc.values, mode='lines', name='BTC Price', line=dict(color='blue')))
        for level in btc_support_levels:
            fig_btc.add_hline(y=level, line=dict(color='green', dash='dash'), annotation_text=f'S: ${level:,.0f}')
        for level in btc_resistance_levels:
            fig_btc.add_hline(y=level, line=dict(color='red', dash='dash'), annotation_text=f'R: ${level:,.0f}')
        fig_btc.update_layout(title='BTC Price with S/R Levels', yaxis_title='Price (USD)')
        st.plotly_chart(fig_btc, use_container_width=True)

    with col2:
        fig_eth = go.Figure()
        fig_eth.add_trace(go.Scatter(x=full_history_eth.index, y=full_history_eth.values, mode='lines', name='ETH Price', line=dict(color='purple')))
        for level in eth_support_levels:
            fig_eth.add_hline(y=level, line=dict(color='green', dash='dash'), annotation_text=f'S: ${level:,.0f}')
        for level in eth_resistance_levels:
            fig_eth.add_hline(y=level, line=dict(color='red', dash='dash'), annotation_text=f'R: ${level:,.0f}')
        fig_eth.update_layout(title='ETH Price with S/R Levels', yaxis_title='Price (USD)')
        st.plotly_chart(fig_eth, use_container_width=True)

else:
    st.error("Failed to load data. Please check connection or try again later.")
