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
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
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
        sigma_val = max(self.sigma[idx], 0.01)  # Prevent zero volatility
        return np.diag([1.0, sigma_val**2])

# Fetch Kraken data
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

    since = int(start_date.timestamp() * 1000)
    until = int(end_date.timestamp() * 1000)
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
    # Fallback: Simulated data
    st.warning("Using fallback simulated data.")
    t = np.linspace(0, 168, 168)  # 7 days in hours
    p0 = 108000  # Approx BTC price based on web data
    prices = p0 + np.cumsum(np.random.normal(0, 1000, 168))
    df = pd.DataFrame({"timestamp": (t * 3600 * 1000).astype(int), "close": prices})
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 50, 500, 100, step=50)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 5, 20, 10, step=5)
epsilon = st.sidebar.slider("Probability Integration Range ($)", 50, 1000, 500, step=50)

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
sigma = np.array([0.02] * len(prices))  # Fallback (2% volatility)
if len(returns) > 5:
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        sigma = res.conditional_volatility / 100
        sigma = np.pad(sigma, (0, 1), mode='edge')
        sigma = np.clip(sigma, 0.01, 0.1)  # Avoid extreme volatility
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
        paths[:, j] = paths[:, j-1] + mu * paths[:, j-1] * dt + sigma[j-1] * paths[:, j-1] * dW[:, j-1]
    return paths, t

with st.spinner("Simulating price paths..."):
    paths, t = simulate_paths(p0, mu, sigma, T, N, n_paths)

# Fokker-Planck equation
def fokker_planck(t, u, sigma, mu, prices, dp):
    sigma_t = sigma[int(np.clip(t / T * (len(sigma) - 1), 0, len(sigma) - 1))]
    d2u = np.zeros_like(u)
    du = np.zeros_like(u)
    if len(u) > 2:
        d2u[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dp**2
        du[1:-1] = (u[2:] - u[:-2]) / (2 * dp)
    result = np.zeros_like(u)
    result[1:-1] = 0.5 * (sigma_t * prices[1:-1])**2 * d2u[1:-1] - mu * prices[1:-1] * du[1:-1]
    return result

# Discretize price space
price_grid = np.linspace(min(prices) * 0.95, max(prices) * 1.05, 200)
dp = price_grid[1] - price_grid[0]
sigma_init = 0.01 * p0  # 1% of initial price
u0 = np.exp(-((price_grid - p0)**2) / (2 * sigma_init**2))
u0 = np.clip(u0, 1e-10, 1e10)
u0 /= np.trapz(u0, price_grid)

# Solve Fokker-Planck
with st.spinner("Solving Fokker-Planck equation..."):
    try:
        sol = solve_ivp(
            fokker_planck,
            [0, T],
            u0,
            method='RK45',
            t_eval=np.linspace(0, T, 50),
            args=(sigma, mu, price_grid, dp),
            rtol=1e-3,
            atol=1e-6
        )
        u = sol.y[:, -1]
        u = np.clip(u, 1e-10, 1e10)
        u /= np.trapz(u, price_grid)
        st.write("Fokker-Planck density sample:", u[:5])
    except Exception as e:
        st.error(f"Fokker-Planck solver failed: {e}. Using fallback density.")
        u = np.exp(-((price_grid - p0)**2) / (2 * (sigma_init * 2)**2))
        u = np.clip(u, 1e-10, 1e10)
        u /= np.trapz(u, price_grid)

# Identify support and resistance levels
du = np.gradient(u, dp)
d2u = np.gradient(du, dp)
support_idx = np.where((du > 0) & (d2u < 0))[0]  # Positive slope, negative curvature
resistance_idx = np.where((du < 0) & (d2u > 0))[0]  # Negative slope, positive curvature
support_levels = price_grid[support_idx]
resistance_levels = price_grid[resistance_idx]
if len(support_levels) == 0 or len(resistance_levels) == 0:
    st.warning("Insufficient distinct levels. Using density peaks.")
    peaks, _ = find_peaks(u, height=0.1 * u.max(), distance=10)
    levels = price_grid[peaks]
    support_levels = levels[levels < np.median(levels)]
    resistance_levels = levels[levels > np.median(levels)]

# Compute hit probabilities
support_probs = []
resistance_probs = []
metric = VolatilityMetric(sigma, t, T)
total_support_prob = 0.0
total_resistance_prob = 0.0

for sr in support_levels:
    mask = (price_grid >= sr - epsilon) & (price_grid <= sr + epsilon)
    if np.any(mask):
        prob = np.trapz(u[mask], price_grid[mask]) * np.sqrt(np.linalg.det(metric.metric_matrix([T, sr])))
        support_probs.append(prob)
        total_support_prob += prob

for rr in resistance_levels:
    mask = (price_grid >= rr - epsilon) & (price_grid <= rr + epsilon)
    if np.any(mask):
        prob = np.trapz(u[mask], price_grid[mask]) * np.sqrt(np.linalg.det(metric.metric_matrix([T, rr])))
        resistance_probs.append(prob)
        total_resistance_prob += prob

# Normalize probabilities
support_probs = [p / total_support_prob if total_support_prob > 0 else 1.0/len(support_levels) for p in support_probs]
resistance_probs = [p / total_resistance_prob if total_resistance_prob > 0 else 1.0/len(resistance_levels) for p in resistance_probs]

# Geodesic
try:
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
    st.error(f"Geodesic computation failed: {e}. Using linear approximation.")
    geodesic_df = pd.DataFrame({
        "Time": t,
        "Price": p0 + (delta_p / T) * t,
        "Path": "Geodesic"
    })

# Prepare data for Altair
path_data = []
for i in range(min(n_paths, n_display_paths)):
    for j in range(N):
        path_data.append({"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"})
path_df = pd.DataFrame(path_data)
if path_df.empty:
    st.error("Path data is empty. Check simulation.")
    st.stop()
plot_df = pd.concat([path_df, geodesic_df])
if plot_df.empty:
    st.error("Plot data is empty. Check geodesic or paths.")
    st.stop()

support_df = pd.DataFrame({"Price": support_levels, "Level": [f"Support_{i}" for i in range(len(support_levels))]})
resistance_df = pd.DataFrame({"Price": resistance_levels, "Level": [f"Resistance_{i}" for i in range(len(resistance_levels))]})

# Altair chart
base = alt.Chart(plot_df).encode(
    x=alt.X("Time:Q", title="Time (hours)", scale=alt.Scale(domain=[0, T])),
    y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(domain=[min(plot_df["Price"]) * 0.95, max(plot_df["Price"]) * 1.05])),
    color=alt.Color("Path:N", legend=None)
)

paths = base.mark_line(opacity=0.2).transform_filter(
    alt.datum.Path != "Geodesic"
)

geodesic = base.mark_line(strokeWidth=3, color="red").transform_filter(
    alt.datum.Path == "Geodesic"
)

support_lines = alt.Chart(support_df).mark_rule(
    strokeDash=[5, 5], color="green", strokeWidth=2
).encode(
    y="Price:Q"
)

resistance_lines = alt.Chart(resistance_df).mark_rule(
    strokeDash=[5, 5], color="red", strokeWidth=2
).encode(
    y="Price:Q"
)

chart = (paths + geodesic + support_lines + resistance_lines).properties(
    title="BTC/USD Price Paths, Geodesic, Support, and Resistance Levels",
    width=800,
    height=400
)

st.altair_chart(chart, use_container_width=True)
st.write("**Support Levels (BTC/USD):**", support_levels)
st.write("**Support Hit Probabilities:**", dict(zip(support_levels, support_probs)))
st.write("**Resistance Levels (BTC/USD):**", resistance_levels)
st.write("**Resistance Hit Probabilities:**", dict(zip(resistance_levels, resistance_probs)))
