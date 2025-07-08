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
import warnings

# --- Global Settings ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("BTC/USD Price Analysis on a Volatility-Weighted Manifold")
st.write("""
This application models the Bitcoin market as a 2D geometric space of (Time, Price), warped by market volatility.
- **Geodesic (Red Line):** The "straightest possible line" or path of least resistance through the curved volatility landscape.
- **S/R Grid:** A grid of probable future support (green) and resistance (red) levels derived from thousands of Monte Carlo simulations.
- **Volume Profile:** A historical map of where trading activity was heaviest, revealing powerful, organic support and resistance zones.
""")

# --- Geometric Modeling Class (Corrected and Final) ---
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        self.dim = 2
        space = geomstats.geometry.euclidean.Euclidean(dim=self.dim)
        super().__init__(space=space)
        self.sigma = sigma; self.t = t; self.T = T
    def metric_matrix(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6)
        return np.diag([1.0, sigma_val**2])
    def christoffel_symbols(self, base_point):
        t_val = base_point[0]; idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-6); eps = 1e-6
        t_plus, t_minus = min(t_val + eps, self.T), max(t_val - eps, 0)
        idx_plus = int(np.clip(t_plus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        idx_minus = int(np.clip(t_minus / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        d_sigma_dt = (self.sigma[idx_plus] - self.sigma[idx_minus]) / (2 * eps)
        gamma = np.zeros((2, 2, 2)); gamma[1, 0, 1] = (1 / sigma_val) * d_sigma_dt
        gamma[1, 1, 0] = gamma[1, 0, 1]; gamma[0, 1, 1] = -sigma_val * d_sigma_dt
        return gamma

# --- Helper Functions (Finalized) ---
@st.cache_data
def fetch_kraken_data(symbol, timeframe, start_date, end_date):
    exchange = ccxt.kraken(); since = int(start_date.timestamp() * 1000)
    limit = int((end_date - start_date).total_seconds() / 3600) + 24
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
        if len(df) >= 10: return df
    except Exception as e: st.warning(f"Could not fetch data for {symbol}: {e}")
    st.error("Failed to fetch recent data. Using simulated data for demonstration.")
    sim_t = pd.date_range(start=start_date, end=end_date, freq='h')
    sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, len(sim_t)))
    sim_df = pd.DataFrame({'datetime': sim_t, 'close': sim_prices, 'volume': np.random.randint(50, 200, len(sim_t))})
    return sim_df

def geodesic_equation(s, y, metric_obj):
    pos, vel = y[:2], y[2:]; gamma = metric_obj.christoffel_symbols(pos)
    accel = -np.einsum('ijk,j,k->i', gamma, vel, vel)
    return np.concatenate([vel, accel])

def simulate_paths(p0, mu, sigma, T, N, n_paths):
    if N < 2: return np.array([[p0]]), np.array([0])
    dt = T / (N - 1); t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N)); paths[:, 0] = p0
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N - 1))
    for j in range(N - 1): paths[:, j + 1] = paths[:, j] * np.exp((mu - 0.5 * sigma[j]**2) * dt + sigma[j] * dW[:, j])
    return paths, t

# --- NEW INTERACTIVE VISUALIZATION FUNCTION ---
def create_interactive_density_chart(price_grid, density, s_levels, r_levels, epsilon):
    fig = go.Figure()
    # Plot the main density curve
    fig.add_trace(go.Scatter(x=price_grid, y=density, mode='lines', name='Probability Density', fill='tozeroy', line_color='lightblue'))
    
    # Add shaded regions for support and resistance
    for level in s_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="green", opacity=0.2, layer="below", line_width=0)
        fig.add_vline(x=level, line_color='green', line_dash='dash')
    for level in r_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="red", opacity=0.2, layer="below", line_width=0)
        fig.add_vline(x=level, line_color='red', line_dash='dash')
        
    fig.update_layout(title="Interactive Probability Distribution with S/R Zones",
                      xaxis_title="Final Price", yaxis_title="Probability Density", showlegend=False)
    return fig

def create_volume_profile_chart(df, n_bins=100):
    price_range = df['close'].max() - df['close'].min()
    bin_size = price_range / n_bins
    df['price_bin'] = (df['close'] // bin_size) * bin_size
    volume_by_price = df.groupby('price_bin')['volume'].sum().reset_index()
    poc = volume_by_price.loc[volume_by_price['volume'].idxmax()]
    fig = go.Figure(go.Bar(y=volume_by_price['price_bin'], x=volume_by_price['volume'], orientation='h', marker_color='lightblue'))
    fig.add_shape(type="line", y0=poc['price_bin'], y1=poc['price_bin'], x0=0, x1=poc['volume'], line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(title="Historical Volume Profile", xaxis_title="Volume Traded", yaxis_title="Price Level (USD)")
    return fig, poc

# --- Main Application Logic ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data to Fetch (Days)", 7, 90, 30)
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 10000, 2000, step=100)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 10, 200, 50, step=10)
epsilon_factor = st.sidebar.slider("Probability Range Factor (for Hit %)", 0.05, 1.0, 0.25, step=0.05)

end_date = pd.Timestamp.now(tz='UTC'); start_date = end_date - pd.Timedelta(days=days_history)
df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 10:
    prices = df['close'].values; times_pd = df['datetime']
    times = (times_pd - times_pd.iloc[0]).dt.total_seconds() / 3600
    T = times.iloc[-1] if not times.empty else 0; N = len(prices); p0 = prices[0]
    returns = 100 * df['close'].pct_change().dropna()
    
    with st.spinner("Fitting GARCH model..."):
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
            sigma = model.conditional_volatility / 100; sigma = np.pad(sigma, (N - len(sigma), 0), mode='edge')
        except Exception:
            sigma = np.full(N, returns.std() / 100 if not returns.empty else 0.02)
    mu = returns.mean() / 100 if not returns.empty else 0

    with st.spinner("Simulating price paths..."):
        paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

    col1, col2 = st.columns([2, 1])
    with col1:
        # --- Main Chart Display ---
        final_prices = paths[:, -1]
        kde = gaussian_kde(final_prices); price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
        u = kde(price_grid); u /= np.trapz(u, price_grid)
        peaks, _ = find_peaks(u, height=0.05 * u.max(), distance=len(price_grid)//25)
        levels = price_grid[peaks]
        
        warning_message = None
        if len(levels) < 4:
            warning_message = "Few distinct peaks found. Using quantiles for S/R grid. This can happen in a strong trend or low-volatility environment."
            support_levels = np.quantile(final_prices, [0.15, 0.40])
            resistance_levels = np.quantile(final_prices, [0.60, 0.85])
        else:
            median_of_peaks = np.median(levels)
            support_levels = levels[levels <= median_of_peaks]; resistance_levels = levels[levels > median_of_peaks]
            if len(resistance_levels) == 0 and len(support_levels) > 1:
                resistance_levels = np.array([support_levels[-1]]); support_levels = support_levels[:-1]
            if len(support_levels) == 0 and len(resistance_levels) > 1:
                support_levels = np.array([resistance_levels[0]]); resistance_levels = resistance_levels[1:]
        
        with st.spinner("Computing geodesic path..."):
            delta_p = prices[-1] - p0
            y0 = np.concatenate([np.array([0.0, p0]), np.array([1.0, delta_p / T if T > 0 else 0.0])])
            sol = solve_ivp(geodesic_equation, [0, T], y0, args=(VolatilityMetric(sigma, t, T),), t_eval=t, rtol=1e-5)
            geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
        
        path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} for i in range(min(n_paths, n_display_paths)) for j in range(N)]
        plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
        support_df = pd.DataFrame({"Price": support_levels}); resistance_df = pd.DataFrame({"Price": resistance_levels})
        base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
        path_lines = base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic")
        geodesic_line = base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic")
        support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q")
        resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
        st.altair_chart((path_lines + geodesic_line + support_lines + resistance_lines).properties(title="Price Paths, Geodesic, and S/R Grid", height=500).interactive(), use_container_width=True)

    with col2:
        # --- Interactive Density Chart and S/R Tables ---
        st.subheader("Projected S/R Probabilities")
        
        # Calculate epsilon based on the slider
        epsilon = epsilon_factor * np.std(final_prices)
        
        # Recalculate hit probabilities based on current epsilon
        metric = VolatilityMetric(sigma, t, T)
        def get_hit_prob(level_list):
            probs = []; total_prob = 0
            for level in level_list:
                mask = (price_grid >= level - epsilon) & (price_grid <= level + epsilon)
                raw_prob = np.trapz(u[mask], price_grid[mask])
                volume_element = np.sqrt(np.abs(np.linalg.det(metric.metric_matrix([T, level]))))
                prob = raw_prob * volume_element
                probs.append(prob); total_prob += prob
            return [p / total_prob for p in probs] if total_prob > 0 else [0] * len(level_list)
        
        support_probs = get_hit_prob(support_levels)
        resistance_probs = get_hit_prob(resistance_levels)
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.write("**Support:**")
            st.dataframe(pd.DataFrame({'Level': support_levels, 'Hit %': support_probs}).style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))
        with sub_col2:
            st.write("**Resistance:**")
            st.dataframe(pd.DataFrame({'Level': resistance_levels, 'Hit %': resistance_probs}).style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))

        with st.expander("Tune Probability Range Factor Visually", expanded=True):
            st.write("""Use the sidebar slider to adjust the 'Probability Range Factor'. Watch how the shaded areas (the 'target zones') expand and contract. Find a value where the zones are distinct but not so narrow that they miss the bulk of the probability curve. **A good value is often between 0.2 and 0.5.**""")
            interactive_density_fig = create_interactive_density_chart(price_grid, u, support_levels, resistance_levels, epsilon)
            st.plotly_chart(interactive_density_fig, use_container_width=True)
            if warning_message: st.info(warning_message)
    
    # --- Volume Profile Section (Full Width) ---
    st.header("Historical Context: Volume-by-Price Analysis")
    st.write("""This chart shows where the market has traded most heavily in the past. High-volume areas act like 'gravity,' often becoming strong support or resistance. **Compare these historical zones with the future S/R levels from the simulation.** A level that is significant in both analyses is a high-conviction zone.""")
    volume_profile_fig, poc = create_volume_profile_chart(df)
    st.plotly_chart(volume_profile_fig, use_container_width=True)
    st.metric("Point of Control (POC): The Most Traded Price", f"${poc['price_bin']:,.2f}")

else:
    st.error("Could not load or process data. Please check parameters or try again later.")
