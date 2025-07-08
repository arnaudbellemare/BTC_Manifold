import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.base import ExpSolver
from geomstats.learning.kmeans import RiemannianKMeans
from arch import arch_model
import warnings
import time
warnings.filterwarnings("ignore")

st.title("BTC/USD Price Analysis on Riemannian Manifold")

# Volatility-weighted metric
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        super().__init__(space=Euclidean(dim=2))
        self.sigma = sigma
        self.t = t
        self.T = T
        self.exp_solver = ExpSolver()

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        return np.diag([1.0, self.sigma[idx]**2])

# Fetch Kraken data
@st.cache_data
def fetch_kraken_data(symbols, timeframe, limit):
    exchange = ccxt.kraken()
    try:
        markets = exchange.load_markets()
        available_symbols = list(markets.keys())
        st.write(f"Available Kraken symbols: {available_symbols[:30]}... (total {len(available_symbols)})")
    except Exception as e:
        st.warning(f"Failed to load Kraken markets: {e}")
        available_symbols = []
    
    since = int((time.time() - 14 * 24 * 3600) * 1000)  # Last 14 days
    st.write(f"Fetching data: since={pd.to_datetime(since, unit='ms')}, limit={limit}, timeframe={timeframe}")
    for symbol in symbols:
        if symbol not in available_symbols:
            st.warning(f"Symbol {symbol} not in Kraken markets")
            continue
        for attempt in range(12):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                if len(df) >= 10 and df['timestamp'].notnull().all() and df['close'].notnull().all() and df['close'].gt(0).all():
                    st.write(f"Success: Fetched {len(df)} data points for {symbol} (attempt {attempt+1})")
                    st.write(f"Sample data: timestamps={df['datetime'].head(3).to_list()}, prices={df['close'].head(3).to_list()}")
                    return df
                else:
                    st.warning(f"Invalid data for {symbol}: len={len(df)}, timestamps_valid={df['timestamp'].notnull().all()}, close_valid={df['close'].notnull().all()}, close_positive={df['close'].gt(0).all()} (attempt {attempt+1})")
            except ccxt.NetworkError as e:
                st.warning(f"Network error for {symbol} (attempt {attempt+1}): {e}")
                time.sleep(15)
            except Exception as e:
                st.warning(f"Error for {symbol} (attempt {attempt+1}): {e}")
    st.error("Failed to fetch valid data from Kraken. Check API status or symbols at https://api.kraken.com/0/public/AssetPairs")
    return None

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 50, 500, 50, step=50)
n_clusters = st.sidebar.slider("Number of K-Means Clusters", 2, 10, 2, step=1)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 5, 20, 10, step=5)

symbols = ['BTC/USD', 'XXBTZUSD', 'XBT/USD', 'BTCUSDT', 'BTC-USD', 'XBTUSDT']
timeframe = '1h'
limit = 10
df = fetch_kraken_data(symbols, timeframe, limit)

# Validate DataFrame
if df is None or df.empty or len(df) < 10 or 'timestamp' not in df or 'close' not in df:
    st.error(f"No valid data fetched from Kraken: df={'None' if df is None else f'len={len(df)}'}")
    st.stop()

# Validate timestamp and close columns
if not df['timestamp'].notnull().all() or not df['close'].notnull().all() or not df['close'].gt(0).all():
    st.error(f"Invalid columns: timestamps_valid={df['timestamp'].notnull().all()}, close_valid={df['close'].notnull().all()}, close_positive={df['close'].gt(0).all()}")
    st.stop()

prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600)

# Validate times and prices
if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(prices)) or not np.all(prices > 0):
    st.error(f"Invalid data: times={len(times)} points, prices={len(prices)} points, times_finite={np.all(np.isfinite(times))}, prices_finite={np.all(np.isfinite(prices))}, prices_positive={np.all(prices > 0)}, times_sample={times[:5] if len(times) > 0 else []}")
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
T = times.iloc[-1]  # Pandas iloc indexing
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

# K-Means clustering for support/resistance
try:
    manifold = Euclidean(dim=2)
    data = np.concatenate([np.vstack([t, paths[i]]).T for i in range(min(n_paths, 5))], axis=0)
    kmeans = RiemannianKMeans(space=manifold, n_clusters=n_clusters, tol=1e-3)
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    support_resistance = cluster_centers[:, 1]  # Price coordinates of centers
except Exception as e:
    st.error(f"K-Means clustering failed: {e}")
    st.stop()

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
        "Price": geodesic_points[:, 1]
    })
except Exception as e:
    st.error(f"Geodesic computation failed: {e}")
    st.stop()

# Matplotlib plot (inspired by Hypersphere tutorial)
fig, ax = plt.subplots(figsize=(10, 6))
# Plot all data points (gray)
ax.scatter(data[:, 0], data[:, 1], color="grey", marker=".", alpha=0.5, label="Data Points")
# Plot clustered points
colors = ['red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'green', 'pink', 'brown']
for i in range(n_clusters):
    ax.scatter(data[labels == i, 0], data[labels == i, 1], color=colors[i % len(colors)], marker=".", alpha=0.7, label=f"Cluster {i}")
# Plot cluster centers
for i, c in enumerate(cluster_centers):
    ax.scatter(c[0], c[1], color=colors[i % len(colors)], marker="*", s=200, label=f"Center {i}")
# Plot simulated paths
for i in range(min(n_paths, n_display_paths)):
    ax.plot(t, paths[i], color="grey", alpha=0.2, label="Simulated Paths" if i == 0 else None)
# Plot geodesic
ax.plot(geodesic_df["Time"], geodesic_df["Price"], color="red", linewidth=2, label="Geodesic")
# Plot support/resistance levels
for i, sr in enumerate(support_resistance):
    ax.axhline(y=sr, color="green", linestyle="--", label=f"Support/Resistance {i}" if i < 2 else None)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("BTC/USD Price")
ax.set_title("BTC/USD Price Paths, Geodesic, and K-Means Support/Resistance Levels")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.write("**K-Means Support/Resistance Levels:**", support_resistance)
