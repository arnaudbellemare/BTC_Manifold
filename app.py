

import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.euclidean
import geomstats
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
- **High Volatility (Yellow areas):** The manifold is 'stretched,' representing periods of high market risk and activity where large price changes are more common.
- **Low Volatility (Dark areas):** The manifold is 'flat,' and movement is 'calmer'.
- **Geodesic (Red Line):** This is the "straightest possible line" through the curved space, representing an idealized path of least resistance according to the volatility landscape.
- **S/R Grid:** The green (support) and red (resistance) lines form a grid of probable future price levels, derived from the Monte Carlo simulation.
""")


# --- Geometric Modeling Class (with geomstats compatibility fix) ---
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        self.dim = 2
        space = geomstats.geometry.euclidean.Euclidean(dim=self.dim)
        super().__init__(space=space)
        self.sigma = sigma
        self.t = t
        self.T = T

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6)
        return np.diag([1.0, sigma_val**2])

    def christoffel_symbols(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6)
        eps = 1e-6
        t_plus, t_minus = min(t_val + eps, self.T), max(t_val - eps, 0)
        idx_plus = int(np.clip(t_plus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        idx_minus = int(np.clip(t_minus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        d_sigma_dt = (self.sigma[idx_plus] - self.sigma[idx_minus]) / (2 * eps)
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
    limit = int((end_date - start_date).total_seconds() / 3600) + 24
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
        if len(df) >= 10:
            st.success(f"Successfully fetched {len(df)} data points for {symbol}.")
            return df
    except Exception as e:
        st.warning(f"Could not fetch data for {symbol}: {e}")
    st.error("Failed to fetch recent data. Using simulated data for demonstration.")
    sim_t = pd.date_range(start=start_date, end=end_date, freq='h')
    sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, len(sim_t)))
    sim_df = pd.DataFrame({'datetime': sim_t, 'close': sim_prices})
    return sim_df

def visualize_manifold(metric, t_grid, p_grid):
    st.subheader("Visualizing the Market Manifold")
    st.write("Heatmap shows the 'cost' of price movement (proportional to σ²). Yellow = high-volatility, 'stretched' regions. Dark = low-volatility, 'flat' regions.")
    SCALING_FACTOR = 10000
    g_pp_values = []
    for t_val in t_grid:
        cost = metric.metric_matrix([t_val, 0])[1, 1]
        scaled_cost = cost * SCALING_FACTOR
        for p_val in p_grid:
            g_pp_values.append({'Time': t_val, 'Price': p_val, 'Cost': scaled_cost})
    g_df = pd.DataFrame(g_pp_values)
    heatmap = alt.Chart(g_df).mark_rect().encode(
        x='Time:Q', y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Cost:Q', scale=alt.Scale(scheme='viridis'),
                        legend=alt.Legend(title=f"Cost (σ² × {SCALING_FACTOR})"))
    ).properties(title="Market Manifold Geometry (Volatility Landscape)")
    return heatmap

def geodesic_equation(s, y, metric_obj):
    pos, vel = y[:2], y[2:]
    gamma = metric_obj.christoffel_symbols(pos)
    accel = -np.einsum('ijk,j,k->i', gamma, vel, vel)
    return np.concatenate([vel, accel])

def simulate_paths(p0, mu, sigma, T, N, n_paths):
    if N < 2: return np.array([[p0]]), np.array([0])
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N)); paths[:, 0] = p0
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N - 1))
    for j in range(N - 1):
        paths[:, j + 1] = paths[:, j] * np.exp((mu - 0.5 * sigma[j]**2) * dt + sigma[j] * dW[:, j])
    return paths, t

# --- Main Application Logic ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data to Fetch (Days)", 7, 90, 30)
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 10000, 2000, step=100)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 10, 200, 50, step=10)
epsilon_factor = st.sidebar.slider("Probability Range Factor (for Hit %)", 0.1, 1.0, 0.25, step=0.05)

end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values
    times_pd = df['datetime']
    times = (times_pd - times_pd.iloc[0]).dt.total_seconds() / 3600
    T = times.iloc[-1]
    N = len(prices)
    p0 = prices[0]
    returns = 100 * df['close'].pct_change().dropna()
    
    with st.spinner("Fitting GARCH model to estimate volatility..."):
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
            sigma = model.conditional_volatility / 100
            sigma = np.pad(sigma, (N - len(sigma), 0), mode='edge')
        except Exception:
            st.warning("GARCH model failed. Using constant volatility.")
            sigma = np.full(N, returns.std() / 100)
    mu = returns.mean() / 100

    with st.spinner("Simulating price paths (Monte Carlo)..."):
        paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

    col1, col2 = st.columns([2, 1])
    with col1:
        final_prices = paths[:, -1]
        kde = gaussian_kde(final_prices)
        price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
        u = kde(price_grid)
        u /= np.trapz(u, price_grid)

        # --- ROBUST S/R GRID CALCULATION ---
        # 1. Find all significant peaks in the final price probability distribution.
        # We use a more sensitive distance to find more potential levels.
        peaks, _ = find_peaks(u, height=0.05 * u.max(), distance=len(price_grid)//25)
        levels = price_grid[peaks]

        if len(levels) >= 2:
            # 2. Robustly separate peaks into S/R using the median of the peaks themselves.
            # This adapts to trends and is better than using the old starting price p0.
            median_of_peaks = np.median(levels)
            support_levels = levels[levels <= median_of_peaks]
            resistance_levels = levels[levels > median_of_peaks]
        else:
            # 3. Fallback to Quantiles: This GUARANTEES a grid of levels.
            st.warning("Few distinct peaks found in distribution. Using quantiles for S/R grid.")
            support_levels = np.quantile(final_prices, [0.15, 0.40])
            resistance_levels = np.quantile(final_prices, [0.60, 0.85])

        # Final check to prevent empty levels if all peaks fall on one side
        if len(resistance_levels) == 0 and len(support_levels) > 1:
             resistance_levels = np.array([support_levels[-1]])
             support_levels = support_levels[:-1]
        if len(support_levels) == 0 and len(resistance_levels) > 1:
             support_levels = np.array([resistance_levels[0]])
             resistance_levels = resistance_levels[1:]
        # --- END OF S/R GRID CALCULATION ---

        metric = VolatilityMetric(sigma, t, T)
        final_std_dev = np.std(final_prices)
        epsilon = epsilon_factor * final_std_dev

        def get_hit_prob(level_list):
            probs = []
            for level in level_list:
                mask = (price_grid >= level - epsilon) & (price_grid <= level + epsilon)
                raw_prob = np.trapz(u[mask], price_grid[mask])
                volume_element = np.sqrt(np.abs(np.linalg.det(metric.metric_matrix([T, level]))))
                probs.append(raw_prob * volume_element)
            total_prob = sum(probs)
            return [p / total_prob for p in probs] if total_prob > 0 else [0] * len(probs)
        
        support_probs = get_hit_prob(support_levels)
        resistance_probs = get_hit_prob(resistance_levels)

        with st.spinner("Computing geodesic path..."):
            delta_p = prices[-1] - p0
            y0 = np.concatenate([np.array([0.0, p0]), np.array([1.0, delta_p / T if T > 0 else 0.0])])
            sol = solve_ivp(geodesic_equation, [0, T], y0, args=(metric,), t_eval=t, rtol=1e-5)
            geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})

        # The Altair charting code now correctly receives multiple S/R levels
        path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} for i in range(min(n_paths, n_display_paths)) for j in range(N)]
        plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
        # These dataframes now contain multiple levels, creating the grid
        support_df = pd.DataFrame({"Price": support_levels})
        resistance_df = pd.DataFrame({"Price": resistance_levels})
        
        base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
        path_lines = base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic")
        geodesic_line = base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic")
        support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
        resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
        main_chart = (path_lines + geodesic_line + support_lines + resistance_lines).properties(title="Price Paths, Geodesic, and S/R Grid", height=500).interactive()
        st.altair_chart(main_chart, use_container_width=True)

    with col2:
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
        else: st.write("No distinct support levels found.")
        st.write("**Resistance Levels (BTC/USD):**")
        if resistance_levels.size > 0:
            st.dataframe(pd.DataFrame({'Level': resistance_levels, 'Hit Probability': resistance_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
        else: st.write("No distinct resistance levels found.")
else:
    st.error("Could not load or process data. Please check parameters or try again later.")
