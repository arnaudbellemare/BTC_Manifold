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
import warnings
from joblib import Parallel, delayed
import streamlit.components.v1 as components

# --- Global Settings ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("BTC/USD Price Analysis on a Volatility-Weighted Manifold")
st.markdown("""
This application models the Bitcoin market as a 2D geometric space (manifold) of (Time, Log-Price), warped by GARCH-derived volatility.  
- **Geodesic (Red Line):** The "straightest" path through the volatility landscape, indicating the market's path of least resistance.  
- **S/R Grid:** Support (green) and resistance (red) levels derived from Monte Carlo simulations, showing probable future price zones.  
- **Volume Profile:** Historical trading activity, highlighting high-volume price levels that act as organic support/resistance.  
*Use the sidebar to adjust parameters and explore the model interactively. Hover over charts for details.*
""")

# --- Geometric Modeling Class ---
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        self.dim = 2
        space = geomstats.geometry.euclidean.Euclidean(dim=self.dim)
        super().__init__(space=space)
        self.sigma = sigma
        self.t = t
        self.T = max(T, 1e-6)  # Prevent division by zero
        # Interpolate sigma for smoother derivatives
        self.sigma_interp = interp1d(t, sigma, bounds_error=False, fill_value=(sigma[0], sigma[-1]))

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        sigma_val = max(self.sigma_interp(t_val), 1e-6)  # Ensure positive definiteness
        return np.diag([1.0, sigma_val**2])

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
    timeframe_seconds = 3600  # 1h timeframe
    all_ohlcv = []
    max_retries = 3
    
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
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed to fetch data after {max_retries} attempts: {e}")
                    break
                continue
    
    progress_bar.empty()
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
        if len(df) >= 10:
            return df
    
    st.error("Failed to fetch data. Using simulated data with volatility clustering.")
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

def create_interactive_density_chart(price_grid, density, s_levels, r_levels, epsilon):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_grid, y=density, mode='lines', name='Probability Density', 
                            fill='tozeroy', line_color='lightblue', hovertemplate='Price: $%{x:,.2f}<br>Density: %{y:.4f}'))
    
    for level in s_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="green", opacity=0.2, 
                      layer="below", line_width=0, annotation_text="Support", annotation_position="top")
        fig.add_vline(x=level, line_color='green', line_dash='dash')
    for level in r_levels:
        fig.add_vrect(x0=level - epsilon, x1=level + epsilon, fillcolor="red", opacity=0.2, 
                      layer="below", line_width=0, annotation_text="Resistance", annotation_position="top")
        fig.add_vline(x=level, line_color='red', line_dash='dash')
        
    fig.update_layout(
        title="Probability Distribution with S/R Zones",
        xaxis_title="Final Price (USD)",
        yaxis_title="Probability Density",
        showlegend=False,
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def create_volume_profile_chart(df, n_bins=100):
    price_range = df['close'].max() - df['close'].min()
    bin_size = price_range / n_bins
    df['price_bin'] = (df['close'] // bin_size) * bin_size
    volume_by_price = df.groupby('price_bin')['volume'].sum().reset_index()
    poc = volume_by_price.loc[volume_by_price['volume'].idxmax()]
    fig = go.Figure(go.Bar(
        y=volume_by_price['price_bin'], 
        x=volume_by_price['volume'], 
        orientation='h', 
        marker_color='lightblue',
        hovertemplate='Price: $%{y:,.2f}<br>Volume: %{x:,.0f}'
    ))
    fig.add_shape(type="line", y0=poc['price_bin'], y1=poc['price_bin'], x0=0, x1=poc['volume'], 
                  line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(
        title="Historical Volume Profile",
        xaxis_title="Volume Traded",
        yaxis_title="Price Level (USD)",
        template="plotly_white"
    )
    return fig, poc

# --- Main Application Logic ---
st.sidebar.header("Model Parameters")
st.sidebar.markdown("Adjust the settings below to customize the analysis. *Hover over labels for details.*")
days_history = st.sidebar.slider(
    "Historical Data (Days)", 7, 90, 30, 
    help="Number of days of historical BTC/USD data to fetch. More data improves model accuracy but increases computation time."
)
n_paths = st.sidebar.slider(
    "Simulated Paths", 500, 10000, 2000, step=100, 
    help="Number of Monte Carlo price paths to simulate. Higher values improve S/R estimates but slow down computation."
)
n_display_paths = st.sidebar.slider(
    "Displayed Paths", 10, 200, 50, step=10, 
    help="Number of simulated paths to show in the main chart. Fewer paths improve visualization performance."
)
epsilon_factor = st.sidebar.slider(
    "Probability Range Factor", 0.05, 1.0, 0.25, step=0.05, 
    help="Controls the width of S/R probability zones. Smaller values create tighter zones; larger values capture more probability."
)
reset_button = st.sidebar.button("Reset to Defaults")
if reset_button:
    days_history, n_paths, n_display_paths, epsilon_factor = 30, 2000, 50, 0.25

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
    
    with st.spinner("Fitting GARCH model..."):
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
            sigma = model.conditional_volatility / 100
            sigma = np.pad(sigma, (N - len(sigma), 0), mode='edge')
        except Exception:
            st.warning("GARCH fitting failed. Using empirical volatility.")
            sigma = np.full(N, returns.std() / 100 if not returns.empty else 0.02)
    mu = returns.mean() / 100 if not returns.empty else 0

    with st.spinner("Simulating price paths..."):
        paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

    col1, col2 = st.columns([2, 1])
    with col1:
        # --- Main Chart ---
        final_prices = paths[:, -1]
        kde = gaussian_kde(final_prices)
        price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
        u = kde(price_grid)
        u /= np.trapz(u, price_grid) + 1e-10  # Ensure normalization
        # Dynamic peak detection
        peak_height = 0.05 * u.max()
        peak_distance = max(10, len(price_grid) // (25 + int(np.std(final_prices) / np.mean(final_prices) * 10)))
        peaks, _ = find_peaks(u, height=peak_height, distance=peak_distance)
        levels = price_grid[peaks]
        
        warning_message = None
        if len(levels) < 4:
            warning_message = "Few distinct peaks found. Using quantiles for S/R grid due to strong trend or low volatility."
            support_levels = np.quantile(final_prices, [0.15, 0.40])
            resistance_levels = np.quantile(final_prices, [0.60, 0.85])
        else:
            median_of_peaks = np.median(levels)
            support_levels = levels[levels <= median_of_peaks]
            resistance_levels = levels[levels > median_of_peaks]
            if len(resistance_levels) == 0 and len(support_levels) > 1:
                resistance_levels = np.array([support_levels[-1]])
                support_levels = support_levels[:-1]
            if len(support_levels) == 0 and len(resistance_levels) > 1:
                support_levels = np.array([resistance_levels[0]])
                resistance_levels = resistance_levels[1:]
        
        with st.spinner("Computing geodesic path..."):
            delta_p = prices[-1] - p0
            # Adaptive initial velocity based on recent trend
            recent_returns = returns[-min(24, len(returns)):].mean() / 100 if len(returns) > 1 else 0
            y0 = np.concatenate([np.array([0.0, p0]), np.array([1.0, delta_p / T + recent_returns])])
            t_eval = np.linspace(0, T, min(N, 100))  # Reduce integration points
            sol = solve_ivp(geodesic_equation, [0, T], y0, args=(VolatilityMetric(sigma, t, T),), 
                           t_eval=t_eval, rtol=1e-5, method='RK23')
            geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
        
        path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} 
                     for i in range(min(n_paths, n_display_paths)) for j in range(N)]
        plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
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
        st.altair_chart(chart, use_container_width=True)

    with col2:
        # --- Density and S/R Tables ---
        st.subheader("Projected S/R Probabilities")
        epsilon = epsilon_factor * np.std(final_prices)
        
        metric = VolatilityMetric(sigma, t, T)
        def get_hit_prob(level_list, price_grid, u, metric, T, epsilon, geodesic_prices):
            probs = []
            total_prob = np.trapz(u, price_grid)
            for level in level_list:
                mask = (price_grid >= level - epsilon) & (price_grid <= level + epsilon)
                raw_prob = np.trapz(u[mask], price_grid[mask])
                metric_mat = metric.metric_matrix([T, level])
                det = max(np.abs(np.linalg.det(metric_mat)), 1e-6)
                volume_element = np.sqrt(det)
                geodesic_price = np.interp(T, geodesic_df["Time"], geodesic_df["Price"])
                geodesic_weight = np.exp(-np.abs(level - geodesic_price) / (2 * epsilon))
                prob = raw_prob * volume_element * geodesic_weight
                probs.append(prob)
            total_level_prob = sum(probs) + 1e-10
            return [p / total_level_prob for p in probs] if total_level_prob > 0 else [0] * len(level_list)
        support_probs = get_hit_prob(support_levels)
        resistance_probs = get_hit_prob(resistance_levels)
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("**Support Levels**")
            st.dataframe(pd.DataFrame({
                'Level': support_levels, 
                'Hit %': support_probs
            }).style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))

        with sub_col2:
            st.markdown("**Resistance Levels**")
            st.dataframe(pd.DataFrame({
                'Level': resistance_levels, 
                'Hit %': resistance_probs
            }).style.format({'Level': '${:,.2f}', 'Hit %': '{:.1%}'}))

        with st.expander("Tune Probability Range Factor", expanded=True):
            st.markdown("""
            Adjust the 'Probability Range Factor' in the sidebar to control the width of S/R zones.  
            - Smaller values create tighter, more precise zones.  
            - Larger values capture more probability but may overlap.  
            **Recommended range: 0.2–0.5.** Watch how the shaded areas respond below.
            """)
            interactive_density_fig = create_interactive_density_chart(price_grid, u, support_levels, resistance_levels, epsilon)
            st.plotly_chart(interactive_density_fig, use_container_width=True)
            if warning_message:
                st.info(warning_message)
    
    # --- Volume Profile ---
    st.header("Historical Context: Volume-by-Price Analysis")
    st.markdown("""
    This chart shows historical trading activity by price level. High-volume areas act as 'gravity,' often serving as strong support or resistance.  
    Compare these zones with the simulated S/R levels above. Overlapping levels indicate high-conviction price zones.  
    *The red dashed line marks the Point of Control (POC), the most traded price.*
    """)
    volume_profile_fig, poc = create_volume_profile_chart(df)
    st.plotly_chart(volume_profile_fig, use_container_width=True)
    st.metric("Point of Control (POC)", f"${poc['price_bin']:,.2f}")

    # --- Export Option ---
    st.markdown("### Export Results")
    export_button = st.button("Download Charts and Data")
    if export_button:
        with st.spinner("Generating exports..."):
            # Export main chart as HTML
            chart.save("main_chart.html")
            with open("main_chart.html", "rb") as f:
                st.download_button("Download Main Chart", f, file_name="main_chart.html")
            # Export density chart
            interactive_density_fig.write_html("density_chart.html")
            with open("density_chart.html", "rb") as f:
                st.download_button("Download Density Chart", f, file_name="density_chart.html")
            # Export volume profile
            volume_profile_fig.write_html("volume_profile.html")
            with open("volume_profile.html", "rb") as f:
                st.download_button("Download Volume Profile", f, file_name="volume_profile.html")
            # Export S/R data
            sr_data = pd.concat([
                pd.DataFrame({'Type': 'Support', 'Level': support_levels, 'Hit %': support_probs}),
                pd.DataFrame({'Type': 'Resistance', 'Level': resistance_levels, 'Hit %': resistance_probs})
            ])
            st.download_button("Download S/R Data", sr_data.to_csv(index=False), file_name="sr_levels.csv")

else:
    st.error("Could not load or process data. Please check parameters, ensure internet connectivity, or try again later.")

# --- Custom CSS for Tooltips ---
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
