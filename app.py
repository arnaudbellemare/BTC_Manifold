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
from dolfinx import fem, mesh, function, io
import ufl
from mpi4py import MPI
import time
import warnings
warnings.filterwarnings("ignore")

st.title("BTC/USD Advanced Riemannian Manifold Analysis (July 1-7, 2025)")

# Advanced Riemannian metric with volatility and Fisher information
class AdvancedRiemannianMetric(RiemannianMetric):
    def __init__(self, sigma, t, T, prices):
        super().__init__(space=Euclidean(dim=2))
        self.sigma = sigma
        self.t = t
        self.T = T
        self.prices = prices
        self.exp_solver = ExpODESolver(space=Euclidean(dim=2), integrator=ScipySolveIVP())

    def metric_matrix(self, base_point):
        t_val, p_val = base_point
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 0.01)
        # Fisher information term from price distribution
        price_dist = np.histogram(self.prices, bins=50, density=True)[0]
        fisher_info = 1.0 / (np.interp(p_val, np.linspace(min(self.prices), max(self.prices), 50), price_dist) + 1e-6)
        return np.diag([1.0, sigma_val**2 * fisher_info])

    def christoffel_symbols(self, base_point):
        t_val, p_val = base_point
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 0.01)
        g = self.metric_matrix(base_point)
        g_inv = np.linalg.inv(g)
        d_sigma = np.gradient(self.sigma, self.t / self.T * (len(self.sigma) - 1))[idx] / (self.T / len(self.sigma))
        gamma = np.zeros((2, 2, 2))
        gamma[1, 1, 0] = 0.5 * g_inv[0, 0] * d_sigma * p_val / sigma_val
        gamma[1, 0, 1] = gamma[1, 1, 0]
        gamma[0, 1, 1] = gamma[1, 1, 0]
        return gamma

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
    st.error("Failed to fetch valid data from Kraken. Using simulated data.")
    t = np.linspace(0, 168, 168)
    p0 = 108000  # Based on web data (~$108,000-$110,150)
    prices = p0 + np.cumsum(np.random.normal(0, 1000, 168)) * (1 + 0.0156 * t / 168)  # 1.56% weekly gain
    df = pd.DataFrame({"timestamp": (t * 3600 * 1000).astype(int), "close": prices})
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 2000, 1000, step=100)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 20, 100, 50, step=10)
epsilon = st.sidebar.slider("Probability Integration Range ($)", 500, 5000, 2000, step=500)

# Fetch data
symbols = ['XBT/USD', 'BTC/USD', 'BTCUSDT', 'XBTUSDT']
timeframe = '1h'
limit = 168
start_date = pd.to_datetime("2025-07-01")
end_date = pd.to_datetime("2025-07-07 23:59:59")
df = fetch_kraken_data(symbols, timeframe, start_date, end_date, limit)

# Validate DataFrame
if df is None or df.empty or len(df) < 10:
    st.error(f"No valid data: df={'None' if df is None else f'len={len(df)}'}")
    st.stop()
if not df['timestamp'].notnull().all() or not df['close'].notnull().all() or not df['close'].gt(0).all():
    st.error(f"Invalid data: timestamps={df['timestamp'].notnull().all()}, close={df['close'].notnull().all()}, positive={df['close'].gt(0).all()}")
    st.stop()

# Prepare data
prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600)
if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(prices)) or not np.all(prices > 0):
    st.error(f"Invalid data: times={len(times)}, prices={len(prices)}, finite={np.all(np.isfinite(times)) and np.all(np.isfinite(prices))}")
    st.stop()

# Heston model parameters
returns = np.diff(prices) / prices[:-1]
v0 = np.var(returns)  # Initial variance
kappa = 0.1  # Mean reversion rate
theta = v0  # Long-term variance
xi = 0.1  # Volatility of variance
rho = -0.3  # Correlation between price and variance
sigma = np.sqrt(v0 * np.ones(len(prices)))  # Initial volatility
if len(returns) > 5:
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        sigma = res.conditional_volatility
        sigma = np.pad(sigma, (0, 1), mode='edge')
    except Exception as e:
        st.warning(f"GARCH failed: {e}. Using Heston parameters.")

p0 = prices[0]
T = times.iloc[-1]
N = len(prices)
mu = np.mean(returns) * N / T if len(returns) > 0 else 0.0

# Simulate paths with Heston model
def simulate_paths(p0, mu, v0, kappa, theta, xi, rho, sigma, T, N, n_paths):
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))
    variances = np.zeros((n_paths, N))
    paths[:, 0] = p0
    variances[:, 0] = v0
    for j in range(1, N):
        z1 = np.random.normal(0, np.sqrt(dt), n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), n_paths)
        variances[:, j] = np.clip(variances[:, j-1] + kappa * (theta - variances[:, j-1]) * dt + xi * np.sqrt(variances[:, j-1]) * z2, 1e-6, np.inf)
        paths[:, j] = paths[:, j-1] + mu * paths[:, j-1] * dt + np.sqrt(variances[:, j-1]) * paths[:, j-1] * z1
    return paths, t, variances

with st.spinner("Simulating paths with Heston model..."):
    paths, t, variances = simulate_paths(p0, mu, v0, kappa, theta, xi, rho, sigma, T, N, n_paths)

# 2D Fokker-Planck on manifold with FEniCS
msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, min(prices) * 0.95]), np.array([T, max(prices) * 1.05])],
                           [100, 200], cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("CG", 2))
u = function.Function(V)
v = function.Function(V)

# Variational form
t_var = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)
metric = AdvancedRiemannianMetric(sigma, t, T, prices)
g = lambda x: metric.metric_matrix([x[0], x[1]])
det_g = ufl.det(ufl.as_matrix([[g(x)[0,0], g(x)[0,1]], [g(x)[1,0], g(x)[1,1]]]))
g_inv = ufl.inv(ufl.as_matrix([[g(x)[0,0], g(x)[0,1]], [g(x)[1,0], g(x)[1,1]])))

# Laplace-Beltrami operator
def laplace_beltrami(u, g_inv):
    return ufl.div(ufl.dot(g_inv, ufl.grad(u)))

# Fokker-Planck variational form
a = ufl.inner(laplace_beltrami(t_var, g_inv), v_test) * ufl.dx
L = ufl.inner(u + 0.5 * (v * p0**2 * laplace_beltrami(u, g_inv) - mu * p0 * ufl.grad(u)[1] +
         xi**2 * v * laplace_beltrami(v, g_inv) - kappa * (theta - v) * ufl.grad(v)[1]), v_test) * ufl.dx

# Initial condition
def initial_condition(x):
    values = np.zeros(x.shape[1])
    values[:] = np.exp(-((x[1] - p0)**2) / (2 * v0 * p0**2)) * np.exp(-((x[0] - 0)**2) / (2 * T**2))
    return values
u.interpolate(initial_condition)

# Time-stepping
dt_fem = T / 2000
t_fem = 0.0
for i in range(2000):
    t_fem += dt_fem
    problem = fem.LinearProblem(a, L, bcs=[])
    u = problem.solve()
    # Update variance (approximate)
    v.vector.set_local(np.clip(v.vector.get_local() + kappa * (theta - v.vector.get_local()) * dt_fem, 1e-6, np.inf))

# Extract 1D density
u_vec = u.vector.get_local()
coords = msh.geometry.x
price_coords = np.unique(coords[:, 1])
u_density = np.zeros(len(price_coords))
for i, p in enumerate(price_coords):
    u_density[i] = np.mean(u_vec[np.where(np.isclose(coords[:, 1], p))[0]])

# Identify support and resistance
du = np.gradient(u_density, price_coords[1] - price_coords[0])
d2u = np.gradient(du, price_coords[1] - price_coords[0])
support_idx = np.where((du > 0) & (d2u < 0) & (u_density > 0.1 * u_density.max()))[0]
resistance_idx = np.where((du < 0) & (d2u > 0) & (u_density > 0.1 * u_density.max()))[0]
support_levels = price_coords[support_idx]
resistance_levels = price_coords[resistance_idx]
if len(support_levels) == 0 or len(resistance_levels) == 0:
    peaks, _ = find_peaks(u_density, height=0.1 * u_density.max(), distance=5)
    levels = price_coords[peaks]
    support_levels = levels[levels < np.median(levels)]
    resistance_levels = levels[levels > np.median(levels)]

# Heat kernel and probabilities
def heat_kernel(t, x, y, metric):
    # Approximate heat kernel using Laplace-Beltrami (simplified for 1D projection)
    dist = metric.dist(x, y)
    return (4 * np.pi * t)**(-0.5) * np.exp(-dist**2 / (4 * t))

support_probs = []
resistance_probs = []
total_support_prob = 0.0
total_resistance_prob = 0.0

for sr in support_levels:
    prob = 0.0
    for p in price_coords:
        if abs(p - sr) <= epsilon:
            prob += heat_kernel(T, [0, p0], [T, p], metric) * np.sqrt(np.linalg.det(metric.metric_matrix([T, p]))) * dp
    support_probs.append(prob)
    total_support_prob += prob

for rr in resistance_levels:
    prob = 0.0
    for p in price_coords:
        if abs(p - rr) <= epsilon:
            prob += heat_kernel(T, [0, p0], [T, p], metric) * np.sqrt(np.linalg.det(metric.metric_matrix([T, p]))) * dp
    resistance_probs.append(prob)
    total_resistance_prob += prob

# Normalize probabilities
support_probs = [p / total_support_prob if total_support_prob > 0 else 1.0 / len(support_levels) for p in support_probs]
resistance_probs = [p / total_resistance_prob if total_resistance_prob > 0 else 1.0 / len(resistance_levels) for p in resistance_probs]

# Geodesic with curvature
try:
    metric = AdvancedRiemannianMetric(sigma, t, T, prices)
    initial_point = np.array([0.0, p0])
    end_point = np.array([T, prices[-1]])
    geodesic = metric.geodesic(initial_point=initial_point, end_point=end_point)
    geodesic_points = geodesic(np.linspace(0, 1, N))
    geodesic_df = pd.DataFrame({
        "Time": geodesic_points[:, 0],
        "Price": geodesic_points[:, 1],
        "Path": "Geodesic"
    })

    # Geodesic distance for trending/reversion
    distances = []
    for i in range(n_paths):
        path_points = np.vstack([t, paths[i]]).T
        dist = metric.dist(path_points, geodesic_points)
        distances.append(np.mean(dist))
    trending = np.array(distances) < np.percentile(distances, 25)
    reversion = np.array(distances) > np.percentile(distances, 75)
except Exception as e:
    st.error(f"Geodesic computation failed: {e}. Using linear approximation.")
    geodesic_df = pd.DataFrame({
        "Time": t,
        "Price": p0 + (prices[-1] - p0) / T * t,
        "Path": "Geodesic"
    })
    distances = [np.mean(np.abs(paths[i] - (p0 + (prices[-1] - p0) / T * t))) for i in range(n_paths)]
    trending = np.array(distances) < np.percentile(distances, 25)
    reversion = np.array(distances) > np.percentile(distances, 75)

# Prepare data for Altair
path_data = []
for i in range(min(n_paths, n_display_paths)):
    for j in range(N):
        path_data.append({"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"})
path_df = pd.DataFrame(path_data)
plot_df = pd.concat([path_df, geodesic_df])
support_df = pd.DataFrame({"Price": support_levels, "Level": [f"Support_{i}" for i in range(len(support_levels))]})
resistance_df = pd.DataFrame({"Price": resistance_levels, "Level": [f"Resistance_{i}" for i in range(len(resistance_levels))]})

# Altair chart
base = alt.Chart(plot_df).encode(
    x=alt.X("Time:Q", title="Time (hours)", scale=alt.Scale(domain=[0, T])),
    y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(domain=[min(plot_df["Price"]) * 0.95, max(plot_df["Price"]) * 1.05])),
    color=alt.Color("Path:N", legend=alt.Legend(title="Paths"))
)

paths = base.mark_line(opacity=0.1).transform_filter(
    alt.datum.Path != "Geodesic"
)

geodesic = base.mark_line(strokeWidth=3, color="red").transform_filter(
    alt.datum.Path == "Geodesic"
)

support_lines = alt.Chart(support_df).mark_rule(
    strokeDash=[5, 5], color="green", strokeWidth=2
).encode(
    y="Price:Q",
    tooltip=["Price", "Level"]
)

resistance_lines = alt.Chart(resistance_df).mark_rule(
    strokeDash=[5, 5], color="red", strokeWidth=2
).encode(
    y="Price:Q",
    tooltip=["Price", "Level"]
)

chart = (paths + geodesic + support_lines + resistance_lines).properties(
    title="BTC/USD Price Paths, Geodesic, Support, and Resistance Levels",
    width=1000,
    height=600
).interactive()

st.altair_chart(chart, use_container_width=True)
st.write("**Support Levels (BTC/USD):**", support_levels)
st.write("**Support Hit Probabilities:**", dict(zip(support_levels, support_probs)))
st.write("**Resistance Levels (BTC/USD):**", resistance_levels)
st.write("**Resistance Hit Probabilities:**", dict(zip(resistance_levels, resistance_probs)))
st.write(f"**Trending Paths:** {np.sum(trending)} (Distance < {np.percentile(distances, 25):.2f})")
st.write(f"**Reversion Paths:** {np.sum(reversion)} (Distance > {np.percentile(distances, 75):.2f})")
