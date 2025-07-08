import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from scipy.signal import find_peaks
from arch import arch_model
from price_model import simulate_paths, compute_density  # Cython
from geomstats.geometry.base import RiemannianMetric
import warnings
import time
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("BTC/USD Price Analysis on Riemannian Manifold")

# Custom volatility-weighted metric
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        super().__init__(dim=2)
        self.sigma = sigma
        self.t = t
        self.T = T

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        return np.diag([1.0, self.sigma[idx]**2])

# Cache data fetching
@st.cache_data
def fetch_kraken_data(symbol, timeframe, since, limit):
    exchange = ccxt.kraken()
    for _ in range(3):  # Retry logic
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except ccxt.NetworkError:
            time.sleep(1)
    st.error("Failed to fetch data from Kraken after retries")
    return None

# User inputs
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 100, 2000, 1000, step=100)
n_bins = st.sidebar.slider("Number of Bins for Density", 20, 100, 50, step=5)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 10, 100, 50, step=10)

# Fetch data
symbol = 'BTC/USD'
timeframe = '1h'
since = ccxt.kraken().parse8601('2025-05-01T00:00:00Z')
limit = 720
df = fetch_kraken_data(symbol, timeframe, since, limit)

if df is None or df.empty:
    st.stop()

# Process data
prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600)

# Estimate volatility with GARCH(1,1)
returns = 100 * np.diff(prices) / prices[:-1]
if len(returns) > 0:
    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    res = model.fit(disp='off')
    sigma = res.conditional_volatility / 100
    sigma = np.pad(sigma, (0, 1), mode='edge')
else:
    st.error("Insufficient data for GARCH model")
    st.stop()

# Parameters
p0 = prices[0]
T = times[-1]
N = len(prices)
mu = np.mean(returns) * N / T / 100 if len(returns) > 0 else 0.0

# Simulate price paths
with st.spinner("Simulating price paths..."):
    paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

# Compute probability density
hist, bins = compute_density(paths, n_bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
density = hist / hist.max()

# Identify support/resistance
peaks, _ = find_peaks(density, height=0.3)
support_resistance = bin_centers[peaks]

# Compute geodesic using geomstats
metric = VolatilityMetric(sigma, t, T)
geodesic = metric.geodesic(
    initial_point=np.array([0.0, p0]),
    initial_tangent_vec=np.array([1.0, mu * T / N])
)
n_points = N
geodesic_points = geodesic(np.linspace(0, 1, n_points))
geodesic_df = pd.DataFrame({
    "Time": geodesic_points[:, 0],
    "Price": geodesic_points[:, 1],
    "Path": "Geodesic"
})

# Create DataFrame for price paths
path_data = []
for i in range(min(n_paths, n_display_paths)):
    for j in range(N):
        path_data.append({"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"})
path_df = pd.DataFrame(path_data)

# Create DataFrame for support/resistance
sr_data = [{"Price": sr, "Level": f"Level_{i}"} for i, sr in enumerate(support_resistance)]
sr_df = pd.DataFrame(sr_data)

# Combine data
plot_df = pd.concat([path_df, geodesic_df])

# Altair chart
base = alt.Chart(plot_df).encode(
    x=alt.X("Time:Q", title="Time (hours)"),
    y=alt.Y("Price:Q", title="BTC/USD Price"),
    color=alt.Color("Path:N", legend=None)
)

paths = base.mark_line(opacity=0.05).transform_filter(
    alt.datum.Path != "Geodesic"
)

geodesic = base.mark_line(strokeWidth=3, color="red").transform_filter(
    alt.datum.Path == "Geodesic"
)

sr_lines = alt.Chart(sr_df).mark_rule(
    strokeDash=[5, 5], color="green", strokeWidth=2
).encode(
    y="Price:Q"
)

chart = (paths + geodesic + sr_lines).properties(
    title="BTC/USD Price Paths, Geodesic, and Support/Resistance Levels",
    width=800,
    height=400
)

# Display chart and results
st.altair_chart(chart, use_container_width=True)
st.write("**Support/Resistance Levels:**", support_resistance)
