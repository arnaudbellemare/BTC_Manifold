import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from scipy.signal import find_peaks
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
import warnings
import time
warnings.filterwarnings("ignore")

st.title("BTC/USD Price Analysis on Riemannian Manifold")

# Volatility-weighted metric
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

# Fetch Kraken data
@st.cache_data
def fetch_kraken_data(symbols, timeframe, limit):
    exchange = ccxt.kraken()
    since = int((time.time() - limit * 3600) * 1000)  # Last 'limit' hours
    st.write(f"Fetching data: since={pd.to_datetime(since, unit='ms')}, limit={limit}")
    for symbol in symbols:
        for attempt in range(7):  # Increased retries
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                if len(df) >= 10 and df['timestamp'].notnull().all() and df['close'].notnull().all():
                    st.write(f"Success: Fetched {len(df)} data points for {symbol} (attempt {attempt+1})")
                    return df
                else:
                    st.warning(f"Invalid data for {symbol}: {len(df)} points (attempt {attempt+1})")
            except ccxt.NetworkError as e:
                st.warning(f"Network error for {symbol} (attempt {attempt+1}): {e}")
                time.sleep(5)
            except Exception as e:
                st.warning(f"Error for {symbol} (attempt {attempt+1}): {e}")
    st.error("Failed to fetch data from Kraken after trying all symbols")
    return None

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 50, 500, 200, step=50)
n_bins = st.sidebar.slider("Number of Bins for Density", 20, 100, 50, step=5)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 5, 20, 10, step=5)

symbols = ['BTC/USD']
timeframe = '1h'
limit = 50  # Reduced further
df = fetch_kraken_data(symbols, timeframe, limit)

if df is None or df.empty or len(df) < 10:
    st.error("No valid data fetched from Kraken. Check API status or try again later.")
    st.stop()

prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600)

# Validate data
if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(prices)):
    st.error(f"Invalid data: times={len(times)} points, prices={len(prices)} points, times_sample={times[:5] if len(times) > 0 else []}")
    st.stop()

# GARCH volatility
returns = 100 * np.diff(prices) / prices[:-1]
if len(returns) > 0:
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        sigma = res.conditional_volatility / 100
        sigma = np.pad(sigma, (0, 1), mode='edge')
    except Exception as e:
        st.error(f"GARCH model failed: {e}")
        st.stop()
else:
    st.error("No returns data for GARCH")
    st.stop()

p0 = prices[0]
T = times[-1]
N = len(prices)
mu = np.mean(returns) * N / T / 100 if len(returns) > 0 else 0.0

# Simulate price paths (pure NumPy)
def simulate_paths(p0, mu, sigma, T, N, n_paths):
    dt = T / N
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))
    paths[:, 0] = p0
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N - 1))
    for j in range(1, N):
        paths[:, j] = paths[:, j-1] + mu * dt + sigma[j-1] * dW[:, j-1]
    return paths, t

with st.spinner("Simulating price paths..."):
    paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

# Compute density
def compute_density(paths, n_bins):
    hist, bins = np.histogram(paths.ravel(), bins=n_bins, density=True)
    return hist, bins

hist, bins = compute_density(paths, n_bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
density = hist / hist.max()

# Support/resistance
peaks, _ = find_peaks(density, height=0.3)
support_resistance = bin_centers[peaks]

# Geodesic with geomstats
try:
    metric = VolatilityMetric(sigma, t, T)
    geodesic = metric.geodesic(
        initial_point=np.array([0.0, p0]),
        initial_tangent_vec=np.array([1.0, mu * 0.001])
    )
    n_points = N
    geodesic_points = geodesic(np.linspace(0, 1, n_points))
    geodesic_df = pd.DataFrame({
        "Time": geodesic_points[:, 0],
        "Price": geodesic_points[:, 1],
        "Path": "Geodesic"
    })
except Exception as e:
    st.error(f"Geodesic computation failed: {e}")
    st.stop()

# DataFrames
path_data = []
for i in range(min(n_paths, n_display_paths)):
    for j in range(N):
        path_data.append({"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"})
path_df = pd.DataFrame(path_data)

sr_data = [{"Price": sr, "Level": f"Level_{i}"} for i, sr in enumerate(support_resistance)]
sr_df = pd.DataFrame(sr_data)

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

st.altair_chart(chart, use_container_width=True)
st.write("**Support/Resistance Levels:**", support_resistance)
