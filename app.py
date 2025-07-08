import streamlit as st
import ccxt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import inv, LinAlgError
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomtimestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            all_data[symbol.split('/')[0]] = df[['close', 'volume']]
        except Exception as e:
            st.error(f"Could not fetch data for {symbol}: {e}")
            return None
    
    if len(all_data) != len(symbols): return None
        
    df_merged = pd.concat(all_data, axis=1)
    df_merged.columns = ['_'.join(col).strip() for col in df_merged.columns.values]
    df_merged.dropna(inplace=True)
    return df_merged

# --- S/R Analysis from Path Distribution ---
def analyze_path_distribution(final_prices, asset_name):
    st.subheader(f"Support & Resistance Analysis for {asset_name}")
    kde = gaussian_kde(final_prices)
    price_grid = np.linspace(final_prices.min(), final_prices.max(), 400)
    density = kde(price_grid)
    peaks, _ = find_peaks(density, height=0.1 * density.max(), distance=10)
    
    if len(peaks) < 2:
        support_levels = [np.quantile(final_prices, 0.25)]
        resistance_levels = [np.quantile(final_prices, 0.75)]
    else:
        levels = price_grid[peaks]
        median_level = np.median(levels)
        support_levels = levels[levels <= median_level]
        resistance_levels = levels[levels > median_level]

    # Plotting is simplified for brevity, your original plotting code is also fine
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_grid, y=density, fill='tozeroy', name='Density'))
    for s in support_levels: fig.add_vline(x=s, line_color='green', line_dash='dash', annotation_text=f'S: ${s:,.2f}')
    for r in resistance_levels: fig.add_vline(x=r, line_color='red', line_dash='dash', annotation_text=f'R: ${r:,.2f}')
    fig.update_layout(title=f'Final Price Density for {asset_name}', xaxis_title='Price', yaxis_title='Density')
    st.plotly_chart(fig, use_container_width=True)

    return support_levels, resistance_levels

# --- Main App Logic ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 10, 90, 30)
rolling_window = st.sidebar.slider("Rolling Window for Volume (Hours)", 12, 168, 72)
n_paths = st.sidebar.slider("Number of Simulated Paths", 500, 5000, 1000, step=100)
projection_hours = st.sidebar.slider("Projection Horizon (Hours)", 12, 168, 72)

df = fetch_and_process_data(days=days_history)

if df is not None:
    # --- MAJOR CHANGE HERE: No more covariance ---
    # We now process volume for each asset individually.
    
    # Calculate rolling mean volume for normalization
    btc_vol_rolling = df['BTC_volume'].rolling(window=rolling_window).mean()
    eth_vol_rolling = df['ETH_volume'].rolling(window=rolling_window).mean()
    
    # The metric component is inversely related to volume.
    # 1 is added for stability (avoids division by zero).
    # We use the mean of the rolling volume as a scaling factor.
    btc_metric_series = 1 / (1 + btc_vol_rolling / btc_vol_rolling.mean())
    eth_metric_series = 1 / (1 + eth_vol_rolling / eth_vol_rolling.mean())

    # Align all data series to a common time index after dropping NaNs from rolling
    common_index = btc_metric_series.dropna().index.intersection(eth_metric_series.dropna().index)
    btc_metric_series = btc_metric_series.loc[common_index]
    eth_metric_series = eth_metric_series.loc[common_index]
    aligned_prices = df.loc[common_index, ['BTC_close', 'ETH_close']]

    if aligned_prices.empty:
        st.error("Data alignment resulted in empty series. Try a smaller rolling window or more data.")
        st.stop()
        
    # --- Geodesic Calculation ---
    try:
        # Instantiate the new, simpler metric class
        metric = AssetVolumeMetric(btc_metric_series, eth_metric_series)
        
        T_total_hours = (aligned_prices.index[-1] - aligned_prices.index[0]).total_seconds() / 3600
        metric.set_time_params(T_total_hours)
        
        initial_point = np.array([0.0, aligned_prices['BTC_close'].iloc[0], aligned_prices['ETH_close'].iloc[0]])
        final_point = np.array([T_total_hours, aligned_prices['BTC_close'].iloc[-1], aligned_prices['ETH_close'].iloc[-1]])
        initial_velocity = (final_point - initial_point) / T_total_hours if T_total_hours > 0 else np.zeros(3)
        y0 = np.concatenate([initial_point, initial_velocity])

        with st.spinner("Calculating geodesic path on the decoupled manifold..."):
            t_eval = np.linspace(0, T_total_hours, len(aligned_prices))
            sol = solve_ivp(geodesic_equation, [0, T_total_hours], y0, args=(metric,), t_eval=t_eval, rtol=1e-4, atol=1e-6)
            geodesic_df = pd.DataFrame(sol.y.T, columns=['time', 'BTC', 'ETH', 'v_t', 'v_btc', 'v_eth'])

    except Exception as e:
        st.error(f"An error occurred during geodesic calculation: {e}")
        st.stop()

    # --- Monte Carlo Simulation (Independent) ---
    with st.spinner("Running Independent Monte Carlo Simulations..."):
        returns = df[['BTC_close', 'ETH_close']].pct_change()
        
        # Calculate mu and sigma for each asset INDEPENDENTLY
        mu_btc = returns['BTC_close'].mean()
        sigma_btc = returns['BTC_close'].std()
        mu_eth = returns['ETH_close'].mean()
        sigma_eth = returns['ETH_close'].std()
        
        p0_btc = aligned_prices['BTC_close'].iloc[-1]
        p0_eth = aligned_prices['ETH_close'].iloc[-1]
        
        dt = 1 # 1-hour steps
        num_steps = projection_hours
        
        # --- MAJOR CHANGE HERE: No Cholesky, just independent random walks ---
        Z_btc = np.random.normal(size=(n_paths, num_steps))
        Z_eth = np.random.normal(size=(n_paths, num_steps))
        
        # Use Geometric Brownian Motion formula
        paths_btc = p0_btc * np.exp(np.cumsum((mu_btc - 0.5 * sigma_btc**2) * dt + sigma_btc * np.sqrt(dt) * Z_btc, axis=1))
        paths_eth = p0_eth * np.exp(np.cumsum((mu_eth - 0.5 * sigma_eth**2) * dt + sigma_eth * np.sqrt(dt) * Z_eth, axis=1))
        
    # --- Visualization Section ---
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=aligned_prices.index, y=aligned_prices['BTC_close'], z=aligned_prices['ETH_close'], mode='lines', line=dict(color='blue', width=4), name='Historical Path'))
    fig.add_trace(go.Scatter3d(x=aligned_prices.index, y=geodesic_df['BTC'], z=geodesic_df['ETH'], mode='lines', line=dict(color='orange', width=6), name='Geodesic Path'))
    
    proj_times = pd.date_range(start=aligned_prices.index[-1], periods=num_steps, freq='h')
    for i in range(min(n_paths, 100)):
        fig.add_trace(go.Scatter3d(x=proj_times, y=paths_btc[i, :], z=paths_eth[i, :], mode='lines', line=dict(color='rgba(128,128,128,0.3)'), showlegend=False))

    fig.update_layout(title='Market Manifold: Historical, Geodesic, and Projected Paths', scene=dict(xaxis_title='Time', yaxis_title='BTC Price (USD)', zaxis_title='ETH Price (USD)'), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        btc_support, btc_resistance = analyze_path_distribution(paths_btc[:, -1], 'BTC')
    with col2:
        eth_support, eth_resistance = analyze_path_distribution(paths_eth[:, -1], 'ETH')
        
    # Your 2D price charts with S/R lines will work the same way as before.

else:
    st.error("Failed to load data. Please check connection or try again later.")
