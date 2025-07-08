import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import Euclidean
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from geomstats.learning.kmeans import RiemannianKMeans
import time
import warnings
warnings.filterwarnings("ignore")

st.title("BTC/USD Price Analysis on Riemannian Manifold (July 1-7, 2025)")

# Volatility-weighted metric
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, t, T):
        super().__init__(space=Euclidean(dim=2))
        self.sigma = sigma
        self.t = t
        self.T = T
        self.exp_solver = ExpODESolver(space=Euclidean(dim=2), integrator=ScipySolveIVP())

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        return np.diag([1.0, self.sigma[idx]**2])

# Fetch Kraken data for last week (July 1-7, 2025)
@st.cache_data
def fetch_kraken_data(symbols, timeframe, start_date, end_date, limit):
    exchange = ccxt.kraken()
    try:
        markets = exchange.load_markets()
        available_symbols = list(markets.keys())
        st.write(f"Available Kraken symbols: {available_symbols[:10]}... (total {len(available_symbols)})")
    except Exception as e:
        st.warning(f"Failed to load Kraken markets: {e}")
        return None

    since = int(start_date.timestamp() * 1000)  # Start of July 1, 2025
    until = int(end_date.timestamp() * 1000)    # End of July 7, 2025
    st.write(f"Fetching data: from {start_date} to {end_date}, timeframe={timeframe}, limit={limit}")

    for symbol in symbols:
        if symbol not in available_symbols:
            st.warning(f"Symbol {symbol} not in Kraken markets")
            continue
        for attempt in range(3):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
                if len(df) >= 10 and df['timestamp'].notnull().all() and df['close'].notnull().all() and df['close'].gt(0).all():
                    st.write(f"Success: Fetched {len(df)} data points for {symbol}")
                    st.write(f"Sample: {df[['datetime', 'close']].head(3).to_dict('records')}")
                    return df
                else:
                    st.warning(f"Invalid data for {symbol}: len={len(df)}")
            except ccxt.NetworkError as e:
                st.warning(f"Network error for {symbol} (attempt {attempt+1}): {e}")
                time.sleep(5)
            except Exception as e:
                st.warning(f"Error for {symbol} (attempt {attempt+1}): {e}")
    st.error("Failed to fetch valid data from Kraken. Check API at https://api.kraken.com/0/public/AssetPairs")
    return None

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 50, 500, 100, step=50)
n_clusters = st.sidebar.slider("Number of K-Means Clusters", 2, 5, 3, step=1)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 5, 20, 10, step=5)

# Fetch data for July 1-7, 2025
symbols = ['XBT/USD', 'BTC/USD', 'BTCUSDT', 'XBTUSDT']
timeframe = '1h'
limit = 168  # 7 days * 24 hours
start_date = pd.to_datetime("2025-07-01")
end_date = pd.to_datetime("2025-07-07 23:59:59")
df = fetch_kraken_data(symbols, timeframe, start_date, end_date, limit)

# Validate DataFrame
if df is None or df.empty or len(df) < 10:
    st.error(f"No valid data fetched: df={'None' if df is None else f'len={len(df)}'}")
    st.stop()

# Validate columns
if not df['timestamp'].notnull().all() or not df['close'].notnull().all() or not df['close'].gt(0).all():
    st.error(f"Invalid data: timestamps_valid={df['timestamp'].notnull().all()}, close_valid={df['close'].notnull().all()}, close_positive={df['close'].gt(0).all()}")
    st.stop()

# Prepare data
prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600)  # Hours
if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(prices)) or not np.all(prices > 0):
    st.error(f"Invalid data: times={len(times)}, prices={len(prices)}, times_finite={np.all(np.isfinite(times))}, prices_finite={np.all(np.isfinite(prices))}")
    st.stop()

# GARCH volatility
returns = 100 * np.diff(prices) / prices[:-1]
sigma = np.array([0.2] * len(prices))  # Fallback constant volatility
if len(returns) > 5:
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        sigma = res.conditional_volatility / 100
        sigma = np.pad(sigma, (0, 1), mode='edge')
    except Exception as e:
        st.warning(f"GARCH failed: {e}. Using constant volatility.")

p0 = prices[0]
T = times.iloc[-1]
N = len(prices)
mu = np.mean(returns) * N / T / 100 if len(returns) > 0 else 0.0

# Simulate price paths
def simulate_paths(p0, mu, sigma, T, N, n_paths):
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))
    paths[:, 0] = p0
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N - 1))
    for j in range(1, N):
        paths[:, j] = paths[:, j-1] + mu * dt + sigma[j-1] * paths[:, j-1] * dW[:, j-1]
    return paths, t

with st.spinner("Simulating price paths..."):
    paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

# K-Means clustering for support/resistance
try:
    manifold = Euclidean(dim=2)
    data = np.concatenate([np.vstack([t, paths[i]]).T for i in range(min(n_paths, 10))], axis=0)
    kmeans = RiemannianKMeans(space=manifold, n_clusters=n_clusters, tol=1e-3)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    support_resistance = np.sort(cluster_centers[:, 1])  # Sort by price
except Exception as e:
    st.error(f"K-Means clustering failed: {e}")
    st.stop()

# Geodesic
try:
    metric = VolatilityMetric(sigma, t, T)
    delta_p = prices[-1] - prices[0]
    geodesic = metric.geodesic(
        initial_point=np.array([0.0, p0]),
        initial_tangent_vec=np.array([T, delta_p])
    )
    geodesic_points = geodesic(np.linspace(0, 1, N))
    geodesic_df = pd.DataFrame({
        "Time": geodesic_points[:, 0],
        "Price": geodesic_points[:, 1],
        "Path": "Geodesic"
    })
except Exception as e:
    st.error(f"Geodesic computation failed: {e}")
    st.stop()

# Prepare data for Altair
path_data = []
for i in range(min(n_paths, n_display_paths)):
    for j in range(N):
        path_data.append({"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"})
path_df = pd.DataFrame(path_data)
plot_df = pd.concat([path_df, geodesic_df])
sr_df = pd.DataFrame({"Price": support_resistance, "Level": [f"Level_{i}" for i in range(len(support_resistance))]})

# Altair chart
base = alt.Chart(plot_df).encode(
    x=alt.X("Time:Q", title="Time (hours)", scale=alt.Scale(domain=[0, T])),
    y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(domain=[min(prices) * 0.95, max(prices) * 1.05])),
    color=alt.Color("Path:N", legend=None)
)

paths = base.mark_line(opacity=0.2).transform_filter(
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
st.write("**Support/Resistance Levels (BTC/USD):**", support_resistance)
