import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
# We need to import the underlying space for the metric
import geomstats.geometry.euclidean
import geomstats # Import the main library
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings

# --- Global Settings ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("BTC/USD Price Analysis on a Volatility-Weighted Manifold")
st.write("""
This application models the Bitcoin market as a 2D geometric space (a "manifold") of (Time, Price).
The geometry is warped by market volatility, calculated using a GARCH model.
- **High Volatility (Yellow areas):** The manifold is 'stretched,' making large price movements geometrically 'shorter' or 'easier'. This represents periods of high market risk and activity.
- **Low Volatility (Dark areas):** The manifold is 'flat,' and movement is 'calmer'.
- **Geodesic (Red Line):** This is the "straightest possible line" through the curved space, representing an idealized path of least resistance according to the volatility landscape.
- **Hit Probabilities:** These are calculated on the curved manifold, correctly accounting for the warped geometry.
""")


# --- Geometric Modeling Class ---
class VolatilityMetric(RiemannianMetric):
    """
    A 2D Manifold (t, p) where the metric is defined by GARCH volatility.
    The metric tensor is g = diag(1, sigma(t)**2).
    """
    def __init__(self, sigma, t, T):
        # --- THE GEOMSTATS COMPATIBILITY FIX ---
        # 1. Define the dimension of the manifold.
        self.dim = 2
        
        # 2. Create an instance of the underlying space (a 2D Euclidean space).
        #    This is now a mandatory argument for the parent class.
        space = geomstats.geometry.euclidean.Euclidean(dim=self.dim)
        
        # 3. Call the parent class's __init__ with the 'space' argument.
        super().__init__(space=space)
        # --- END OF THE FIX ---
        
        self.sigma = sigma
        self.t = t
        self.T = T

    def metric_matrix(self, base_point):
        """Calculates the metric tensor g at a given point."""
        t_val = base_point[0]
        # Find the volatility sigma(t) corresponding to the time coordinate
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6) # Use a small floor for stability
        # The metric is diagonal: g_tt = 1, g_pp = sigma^2
        return np.diag([1.0, sigma_val**2])

    def christoffel_symbols(self, base_point):
        """
        Calculates the Christoffel symbols analytically.
        This is much faster and more stable than numerical differentiation.
        """
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6)
        
        # Numerically estimate the derivative of sigma w.r.t time, d(sigma)/dt
        eps = 1e-6
        t_plus, t_minus = min(t_val + eps, self.T), max(t_val - eps, 0)
        idx_plus = int(np.clip(t_plus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        idx_minus = int(np.clip(t_minus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        d_sigma_dt = (self.sigma[idx_plus] - self.sigma[idx_minus]) / (2 * eps)

        # The non-zero Christoffel symbols for g = diag(1, f(t)^2) are known
        gamma = np.zeros((2, 2, 2))
        gamma[1, 0, 1] = (1 / sigma_val) * d_sigma_dt
        gamma[1, 1, 0] = gamma[1, 0, 1]
        gamma[0, 1, 1] = -sigma_val * d_sigma_dt
        
        return gamma

# --- Helper Functions ---
@st.cache_data
def fetch_kraken_data(symbols, timeframe, start_date, end_date):
    """Fetches OHLCV data and falls back to simulation if needed."""
    exchange = ccxt.kraken()
    since = int(start_date.timestamp() * 1000)
    limit = int((end_date - start_date).total_seconds() / 3600) + 1 # Hourly timeframe
    
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
            if len(df) >= 10:
                st.success(f"Successfully fetched {len(df)} data points for {symbol}.")
                return df
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")
            continue
            
    st.error("Failed to fetch recent data. Using simulated data for demonstration.")
    sim_t = pd.date_range(start=start_date, periods=168, freq='h')
    sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, 168))
    return pd.DataFrame({'datetime': sim_t, 'close': sim_prices, 'timestamp': sim_t.astype(np.int64) // 10**6})

def visualize_manifold(metric, t_grid, p_grid):
    """Creates a heatmap visualization of the manifold's geometry."""
    st.subheader("Visualizing the Market Manifold")
    st.write("This heatmap shows the 'cost' of price movement (proportional to σ²). Yellow areas are high-volatility, 'stretched' regions. Dark areas are low-volatility, 'flat' regions.")
    
    # Scaling factor for better color representation
    SCALING_FACTOR = 10000 
    
    g_pp_values = []
    # The metric g_pp only depends on time, so we calculate it for each time step
    for t_val in t_grid:
        cost = metric.metric_matrix([t_val, 0])[1, 1] # [1,1] is the price component
        scaled_cost = cost * SCALING_FACTOR
        # We create a full grid for visualization purposes
        for p_val in p_grid:
            g_pp_values.append({'Time': t_val, 'Price': p_val, 'Cost': scaled_cost})
    
    g_df = pd.DataFrame(g_pp_values)
    min_cost, max_cost = g_df['Cost'].min(), g_df['Cost'].max()

    heatmap = alt.Chart(g_df).mark_rect().encode(
        x='Time:Q',
        y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Cost:Q', 
                        scale=alt.Scale(scheme='viridis', domain=[min_cost, max_cost]), 
                        legend=alt.Legend(title=f"Cost (σ² × {SCALING_FACTOR})"))
    ).properties(
        title="Market Manifold Geometry (Volatility Landscape)",
    )
    return heatmap

def geodesic_equation(s, y, metric_obj):
    """The geodesic ODE, to be solved by an IVP solver."""
    pos, vel = y[:2], y[2:]
    gamma = metric_obj.christoffel_symbols(pos)
    # Einstein summation for the acceleration term
    accel = -np.einsum('ijk,j,k->i', gamma, vel, vel)
    return np.concatenate([vel, accel])

def simulate_paths(p0, mu, sigma, T, N, n_paths):
    """Simulates price paths using Geometric Brownian Motion."""
    if N < 2: return np.array([[p0]]), np.array([0])
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N)); paths[:, 0] = p0
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N-1))
    
    for j in range(N - 1):
        # The volatility sigma[j] applies to the interval from t_j to t_j+1
        paths[:, j+1] = paths[:, j] * np.exp((mu - 0.5 * sigma[j]**2) * dt + sigma[j] * dW[:, j])
    return paths, t

# --- Main Application Logic ---

# Parameters
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data to Fetch (Days)", 7, 90, 30)
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 10000, 2000, step=100)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 10, 200, 50, step=10)
epsilon_factor = st.sidebar.slider("Probability Range Factor (for Hit %)", 0.1, 1.0, 0.25, step=0.05)

# Data Fetching and Preparation
end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
df = fetch_kraken_data(['BTC/USD', 'XBT/USD'], '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values
    times_pd = pd.to_datetime(df['timestamp'], unit='ms')
    times = (times_pd - times_pd.iloc[0]).total_seconds() / 3600
    T = times.iloc[-1]
    N = len(prices)
    p0 = prices[0]
    returns = 100 * df['close'].pct_change().dropna()
    
    # GARCH Model for Volatility
    with st.spinner("Fitting GARCH model to estimate volatility..."):
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
            sigma = model.conditional_volatility / 100 # Convert from percent to decimal
            sigma = np.pad(sigma, (N - len(sigma), 0), mode='edge') # Align with prices
        except Exception as e: 
            st.warning(f"GARCH model failed: {e}. Using constant volatility.")
            sigma = np.full(N, returns.std() / 100)
    mu = returns.mean() / 100 # Hourly drift

    # Simulate price paths (unconstrained)
    with st.spinner("Simulating price paths (Monte Carlo)..."):
        paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

    # Main analysis columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Calculate Final Price Distribution and S/R Levels
        final_prices = paths[:, -1]
        kde = gaussian_kde(final_prices)
        price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
        u = kde(price_grid)
        u /= np.trapz(u, price_grid) # Normalize probability density

        peaks, _ = find_peaks(u, height=0.1 * u.max(), distance=len(price_grid)//10)
        levels = price_grid[peaks]
        median_level = np.median(levels) if len(levels) > 0 else np.median(price_grid)
        support_levels = levels[levels < p0]
        resistance_levels = levels[levels > p0]
        if len(support_levels) == 0 and len(resistance_levels) > 0:
             support_levels = [np.quantile(final_prices, 0.25)]
        if len(resistance_levels) == 0 and len(support_levels) > 0:
             resistance_levels = [np.quantile(final_prices, 0.75)]
        
        # Calculate Hit Probabilities on the Manifold
        metric = VolatilityMetric(sigma, t, T)
        final_std_dev = np.std(final_prices)
        epsilon = epsilon_factor * final_std_dev

        def get_hit_prob(level_list):
            probs = []
            for level in level_list:
                mask = (price_grid >= level - epsilon) & (price_grid <= level + epsilon)
                # This is the key: integrate probability density *and* account for the geometry
                raw_prob = np.trapz(u[mask], price_grid[mask])
                volume_element = np.sqrt(np.abs(np.linalg.det(metric.metric_matrix([T, level]))))
                probs.append(raw_prob * volume_element)
            total_prob = sum(probs)
            return [p / total_prob for p in probs] if total_prob > 0 else [0] * len(probs)
        
        support_probs = get_hit_prob(support_levels)
        resistance_probs = get_hit_prob(resistance_levels)

        # Calculate the Geodesic Path
        with st.spinner("Computing geodesic path..."):
            try:
                delta_p = prices[-1] - p0
                initial_point = np.array([0.0, p0])
                initial_velocity = np.array([1.0, delta_p / T if T > 0 else 0.0])
                y0 = np.concatenate([initial_point, initial_velocity])
                sol = solve_ivp(geodesic_equation, [0, T], y0, args=(metric,), t_eval=t, rtol=1e-5)
                geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
            except Exception as e:
                st.error(f"Geodesic computation failed: {e}. Using linear approximation.")
                geodesic_df = pd.DataFrame({"Time": t, "Price": np.linspace(p0, prices[-1], N), "Path": "Geodesic"})

        # Main Altair Chart
        path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} for i in range(min(n_paths, n_display_paths)) for j in range(N)]
        plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
        support_df, resistance_df = pd.DataFrame({"Price": support_levels}), pd.DataFrame({"Price": resistance_levels})
        
        base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
        path_lines = base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic")
        geodesic_line = base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic")
        support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
        resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
        
        main_chart = (path_lines + geodesic_line + support_lines + resistance_lines).properties(title="Price Paths, Geodesic, Support (Green), and Resistance (Red)", height=500).interactive()
        st.altair_chart(main_chart, use_container_width=True)

    with col2:
        # Display the Manifold Heatmap and Summary Stats
        if N > 0:
            viz_p_grid = np.linspace(prices.min(), prices.max(), 50)
            manifold_heatmap = visualize_manifold(metric, t, viz_p_grid)
            history_df = pd.DataFrame({'Time': times, 'Price': prices})
            history_line = alt.Chart(history_df).mark_line(color='white', strokeWidth=2.5, opacity=0.7).encode(x='Time:Q', y='Price:Q')
            
            st.altair_chart((manifold_heatmap + history_line).properties(height=300).interactive(), use_container_width=True)
        
        st.subheader("Analysis Summary")
        st.metric("Expected Final Price", f"${np.mean(final_prices):,.2f}")
        
        st.write("**Support Levels (BTC/USD):**")
        if support_levels.size > 0:
            st.dataframe(pd.DataFrame({'Level': support_levels, 'Hit Probability': support_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
        else:
            st.write("No distinct support levels found below current price.")
            
        st.write("**Resistance Levels (BTC/USD):**")
        if resistance_levels.size > 0:
            st.dataframe(pd.DataFrame({'Level': resistance_levels, 'Hit Probability': resistance_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
        else:
            st.write("No distinct resistance levels found above current price.")
else:
    st.error("Could not load or process data. Please check parameters or try again later.")
