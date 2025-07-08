import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import time
import warnings
warnings.filterwarnings("ignore")

st.title("BTC/USD Price Analysis on a Volatility-Weighted Manifold")

# Volatility-weighted metric
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        # --- THE DEFINITIVE FIX FOR GEOMSTATS COMPATIBILITY ---
        # The parent constructor is NOT called. We manually set the required attributes.
        # This works on all versions of the library.
        self.dim = 2
        
        self.sigma = sigma
        self.t = t
        self.T = T
        self.exp_solver = ExpODESolver(self)

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        # Ensure index is within bounds for the sigma array
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 0.01)
        return np.diag([1.0, sigma_val**2])

# Fetch Kraken data (robust version)
@st.cache_data
def fetch_kraken_data(symbols, timeframe, start_date, end_date, limit):
    exchange = ccxt.kraken()
    try: exchange.load_markets()
    except Exception as e:
        st.warning(f"Failed to load Kraken markets: {e}")
        return None
    since = int(start_date.timestamp() * 1000)
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
            if len(df) >= 10: return df
        except Exception: continue
    st.error("Failed to fetch valid data for July 1-7, 2025. Using simulated data.")
    sim_t = pd.date_range(start=start_date, periods=168, freq='h')
    sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, 168))
    return pd.DataFrame({'datetime': sim_t, 'close': sim_prices, 'timestamp': sim_t.astype(np.int64) // 10**6})

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 5000, 2000, step=100)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 10, 200, 50, step=10)
epsilon_factor = st.sidebar.slider("Probability Range Factor", 0.1, 1.0, 0.25, step=0.05)
st.sidebar.info("Note: Price simulation is capped at $200,000.")

# Fetch data for July 1-7, 2025
df = fetch_kraken_data(['BTC/USD', 'XBT/USD'], '1h', pd.to_datetime("2025-07-01"), pd.to_datetime("2025-07-07 23:59:59"), 168)

# Data Preparation
prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600) if not df.empty else np.array([])
T = times.iloc[-1] if len(times) > 0 else 168
N = len(prices)
p0 = prices[0] if N > 0 else 70000
returns = 100 * np.diff(prices) / prices[:-1] if N > 1 else np.array([])
mu = np.mean(returns) * N / T / 100 if len(returns) > 0 else 0.0
sigma = np.array([0.02] * N) if N > 0 else np.array([])
if len(returns) > 5:
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        sigma = res.conditional_volatility / 100
        sigma = np.pad(sigma, (1, 0), mode='edge')
    except Exception as e: st.warning(f"GARCH failed: {e}. Using constant volatility.")

# Simulate price paths (GBM from working script)
def simulate_paths(p0, mu, sigma, T, N, n_paths, price_cap=200000.0):
    if N == 0: return np.array([[p0]]), np.array([0])
    dt = T / (N - 1) if N > 1 else T
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N)); paths[:, 0] = p0
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N))
    for j in range(1, N):
        paths[:, j] = paths[:, j-1] + mu * paths[:, j-1] * dt + sigma[j-1] * paths[:, j-1] * dW[:, j-1]
        paths[:, j] = np.clip(paths[:, j], 0, price_cap)
    return paths, t

with st.spinner("Simulating price paths (Monte Carlo)..."):
    paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

# --- ROBUST FOKKER-PLANCK SOLUTION VIA KDE ---
with st.spinner("Constructing Fokker-Planck solution via KDE..."):
    final_prices = paths[:, -1]
    kde = gaussian_kde(final_prices)
    price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
    u = kde(price_grid)
    u /= np.trapz(u, price_grid)
    dp = price_grid[1] - price_grid[0]

# Identify support and resistance levels
du = np.gradient(u, dp)
d2u = np.gradient(du, dp)
support_idx = np.where((du > 0) & (d2u < 0) & (u > 0.05 * u.max()))[0]
resistance_idx = np.where((du < 0) & (d2u > 0) & (u > 0.05 * u.max()))[0]
support_levels = price_grid[support_idx]
resistance_levels = price_grid[resistance_idx]
if len(support_levels) == 0 or len(resistance_levels) == 0:
    st.warning("Insufficient distinct levels. Using density peaks as fallback.")
    peaks, _ = find_peaks(u, height=0.1 * u.max(), distance=10)
    levels = price_grid[peaks]
    median_level = np.median(levels) if len(levels) > 0 else np.median(price_grid)
    support_levels = levels[levels < median_level]
    resistance_levels = levels[levels > median_level]

# Compute hit probabilities
support_probs, resistance_probs = [], []
metric = VolatilityMetric(sigma, t, T)
total_support_prob, total_resistance_prob = 0.0, 0.0
final_std_dev = np.std(final_prices)
epsilon = epsilon_factor * final_std_dev

for sr in support_levels:
    mask = (price_grid >= sr - epsilon) & (price_grid <= sr + epsilon)
    prob = np.trapz(u[mask], price_grid[mask]) * np.sqrt(np.abs(np.linalg.det(metric.metric_matrix([T, sr]))))
    support_probs.append(prob)
    total_support_prob += prob
for rr in resistance_levels:
    mask = (price_grid >= rr - epsilon) & (price_grid <= rr + epsilon)
    prob = np.trapz(u[mask], price_grid[mask]) * np.sqrt(np.abs(np.linalg.det(metric.metric_matrix([T, rr]))))
    resistance_probs.append(prob)
    total_resistance_prob += prob

if total_support_prob > 0: support_probs = [p / total_support_prob for p in support_probs]
if total_resistance_prob > 0: resistance_probs = [p / total_resistance_prob for p in resistance_probs]

# Geodesic
try:
    delta_p = prices[-1] - p0 if N > 0 else 0
    geodesic_func = metric.geodesic(initial_point=np.array([0.0, p0]), initial_tangent_vec=np.array([T, delta_p]))
    geodesic_points = geodesic_func(np.linspace(0, 1, N))
    geodesic_df = pd.DataFrame({"Time": geodesic_points[:, 0], "Price": geodesic_points[:, 1], "Path": "Geodesic"})
except Exception as e:
    st.error(f"Geodesic computation failed: {e}. Using linear approximation.")
    geodesic_df = pd.DataFrame({"Time": t, "Price": p0 + (delta_p / T if T > 0 else 0) * t, "Path": "Geodesic"})

# Altair chart
path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} for i in range(min(n_paths, n_display_paths)) for j in range(N)]
plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
support_df = pd.DataFrame({"Price": support_levels})
resistance_df = pd.DataFrame({"Price": resistance_levels})
base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
path_lines = base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic")
geodesic = base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic")
support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=2, strokeDash=[5, 5]).encode(y="Price:Q")
resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=2, strokeDash=[5, 5]).encode(y="Price:Q")
chart = (path_lines + geodesic + support_lines + resistance_lines).properties(title="Price Paths, Geodesic, Support (Green), and Resistance (Red)", width=800, height=400).interactive()

st.altair_chart(chart, use_container_width=True)
st.write(f"**Expected Final Price:** ${np.mean(final_prices):,.2f}")
st.write("**Support Levels (BTC/USD):**")
if support_levels.size > 0:
    st.dataframe(pd.DataFrame({'Level': support_levels, 'Hit Probability': support_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
else:
    st.write("No distinct support levels found.")
st.write("**Resistance Levels (BTC/USD):**")
if resistance_levels.size > 0:
    st.dataframe(pd.DataFrame({'Level': resistance_levels, 'Hit Probability': resistance_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
else:
    st.write("No distinct resistance levels found.")
