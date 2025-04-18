import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
import pywt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import umap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

st.set_page_config(
    page_title="Financial Time Resonance Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------- Utility Functions -------------------

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess financial data from CSV file"""
    if uploaded_file is None:
        st.warning("Please upload a file to proceed.")
        return None

    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.fillna(method='ffill')
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df = df.dropna()
    return df


def calculate_features(data):
    """Calculate additional features for resonance analysis"""
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['close_norm'] = normalize_series(data['Close'])
    features['returns'] = data['Returns']
    features['log_returns'] = np.log1p(data['Returns'])
    features['volatility'] = data['Volatility']
    features['price_range'] = data['Price_Range']
    
    # Moving averages and trends
    features['sma20_close_ratio'] = data['Close'] / data['SMA_20']
    features['sma_ratio'] = data['SMA_20'] / data['SMA_50']
    
    # Volume-based features
    features['volume_norm'] = normalize_series(data['Volume'])
    features['volume_change'] = data['Volume_Change']
    
    # Momentum indicators
    features['momentum_5d'] = data['Close'].pct_change(5)
    features['momentum_10d'] = data['Close'].pct_change(10)
    
    return features

def normalize_series(series):
    """Min-max normalize a series to 0-1 range"""
    return (series - series.min()) / (series.max() - series.min())

def segment_data(data, window_size):
    """Segment the data into windows of specific size"""
    segments = []
    for i in range(0, len(data) - window_size + 1):
        segments.append(data.iloc[i:i+window_size])
    return segments

def calculate_similarity(segment1, segment2, method='pearson'):
    """Calculate similarity between two segments using specified method"""
    if method == 'pearson':
        correlation, _ = pearsonr(segment1, segment2)
        return abs(correlation)  # Use absolute correlation as similarity measure
    elif method == 'dtw':
        distance, _ = fastdtw(segment1, segment2, dist=euclidean)
        # Convert distance to similarity (inverse relationship)
        return 1 / (1 + distance)
    elif method == 'euclidean':
        distance = np.sqrt(np.sum((segment1 - segment2) ** 2))
        return 1 / (1 + distance)
    else:
        return 0

def compute_wavelet_transform(data, wavelet='db4', level=5):
    """Compute wavelet transform for multi-scale analysis"""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def generate_decompositions(data, column='close_norm'):
    """Generate multi-timeframe decompositions of the data"""
    decompositions = {}
    
    # Daily patterns
    decompositions['daily'] = data[column].values
    
    # Weekly pattern (resample to weekly)
    weekly = data[column].resample('W').mean()
    decompositions['weekly'] = weekly.values
    
    # Monthly pattern
    monthly = data[column].resample('M').mean()
    decompositions['monthly'] = monthly.values
    
    # Quarterly pattern
    quarterly = data[column].resample('Q').mean()
    decompositions['quarterly'] = quarterly.values
    
    # Wavelet decomposition for fractal patterns
    wavelet = compute_wavelet_transform(data[column].values)
    for i, coeff in enumerate(wavelet):
        decompositions[f'wavelet_level_{i}'] = coeff
    
    return decompositions

def find_resonances(data, current_window, window_size, n_results=5):
    """
    Find historical periods that resonate with the current window
    based on multiple features and timeframes
    """
    features = calculate_features(data)
    
    # Get current window features
    if len(data) < window_size:
        return None, None
    
    current_features = features.iloc[-window_size:]
    
    # Segment historical data
    historical_segments = []
    dates = []
    
    for i in range(0, len(features) - window_size - window_size):  # Avoid overlap with current window
        segment = features.iloc[i:i+window_size]
        historical_segments.append(segment)
        dates.append(segment.index[0])
    
    # Calculate resonance scores
    similarity_scores = []
    
    # Features to compare
    feature_columns = ['close_norm', 'returns', 'volatility', 'volume_norm', 'price_range']
    
    for segment in historical_segments:
        segment_scores = []
        
        for feature in feature_columns:
            # Calculate similarity for this feature
            sim = calculate_similarity(
                current_features[feature].values, 
                segment[feature].values
            )
            segment_scores.append(sim)
        
        # Average score across all features
        avg_score = np.mean(segment_scores)
        similarity_scores.append(avg_score)
    
    # Get top resonance periods
    results_df = pd.DataFrame({
        'Start_Date': dates,
        'Resonance_Score': similarity_scores
    })
    
    # Sort by resonance score in descending order
    results_df = results_df.sort_values('Resonance_Score', ascending=False)
    
    # Get top n results
    top_resonances = results_df.head(n_results)
    
    # Extract the actual data segments for the top resonances
    top_segments = []
    for date in top_resonances['Start_Date']:
        idx = data.index.get_indexer([date])[0]
        segment = data.iloc[idx:idx+window_size]
        top_segments.append(segment)
    
    return top_resonances, top_segments

def calculate_future_trajectories(data, resonances, top_segments, window_size, forecast_horizon=30):
    """Calculate potential future trajectories based on historical resonances"""
    trajectories = []
    
    # Calculate the average trajectory
    avg_trajectory = np.zeros(forecast_horizon)
    
    for i, segment in enumerate(top_segments):
        # If we have data after this segment, use it as a potential future trajectory
        start_idx = data.index.get_indexer([segment.index[0]])[0]
        
        if start_idx + window_size + forecast_horizon <= len(data):
            # Extract the future trajectory after this segment
            future_segment = data.iloc[start_idx+window_size:start_idx+window_size+forecast_horizon]
            
            # Normalize the future trajectory to start at the current price
            norm_factor = data['Close'].iloc[-1] / future_segment['Close'].iloc[0]
            future_trajectory = future_segment['Close'].values * norm_factor
            
            # Assign a weight based on the resonance score
            weight = resonances['Resonance_Score'].iloc[i]
            
            trajectories.append({
                'trajectory': future_trajectory,
                'start_date': segment.index[0],
                'resonance_score': weight
            })
            
            # Add weighted contribution to average trajectory
            avg_trajectory += future_trajectory * weight
    
    # Normalize the average trajectory
    if trajectories:
        total_weight = resonances['Resonance_Score'].sum()
        avg_trajectory /= total_weight
    
    return trajectories, avg_trajectory

def visualize_resonances_3d(current_data, historical_segments, scores):
    """Create 3D visualization of resonance relationships"""
    # Extract features for dimensionality reduction
    features_list = []
    
    # Add current data
    current_features = np.concatenate([
        normalize_series(current_data['Close']).values,
        current_data['Volatility'].values,
        normalize_series(current_data['Volume']).values
    ])
    features_list.append(current_features)
    
    # Add historical segments
    for segment in historical_segments:
        segment_features = np.concatenate([
            normalize_series(segment['Close']).values,
            segment['Volatility'].values,
            normalize_series(segment['Volume']).values
        ])
        features_list.append(segment_features)
    
    # Perform dimensionality reduction with UMAP
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(features_list)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Plot historical segments
    for i in range(1, len(embedding)):
        fig.add_trace(go.Scatter3d(
            x=[embedding[i, 0]],
            y=[embedding[i, 1]],
            z=[embedding[i, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color=scores[i-1],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Resonance Score')
            ),
            text=f"Start Date: {historical_segments[i-1].index[0].strftime('%Y-%m-%d')}<br>Score: {scores[i-1]:.4f}",
            hoverinfo='text',
            name=f"Historical {i}"
        ))
    
    # Plot current data point with larger marker
    fig.add_trace(go.Scatter3d(
        x=[embedding[0, 0]],
        y=[embedding[0, 1]],
        z=[embedding[0, 2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        text="Current Market Position",
        hoverinfo='text',
        name="Current"
    ))
    
    # Add lines connecting current position to top resonances
    for i in range(1, min(4, len(embedding))):
        fig.add_trace(go.Scatter3d(
            x=[embedding[0, 0], embedding[i, 0]],
            y=[embedding[0, 1], embedding[i, 1]],
            z=[embedding[0, 2], embedding[i, 2]],
            mode='lines',
            line=dict(
                color='rgba(100, 100, 100, 0.4)',
                width=2
            ),
            hoverinfo='none',
            showlegend=False
        ))
    
    fig.update_layout(
        title="3D Resonance Field Map",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def compare_patterns(current_segment, historical_segment):
    """Create comparison visualization between current and historical patterns"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Price Comparison", "Volume Comparison"),
        vertical_spacing=0.15
    )
    
    # Normalize both segments to start at 100 for better comparison
    current_norm = 100 * current_segment['Close'] / current_segment['Close'].iloc[0]
    historical_norm = 100 * historical_segment['Close'] / historical_segment['Close'].iloc[0]
    
    # Create date ranges for x-axis
    current_dates = np.arange(len(current_segment))
    historical_dates = np.arange(len(historical_segment))
    
    # Price comparison
    fig.add_trace(
        go.Scatter(
            x=current_dates,
            y=current_norm,
            mode='lines',
            name='Current',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=historical_dates,
            y=historical_norm,
            mode='lines',
            name='Historical',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Volume comparison
    current_vol_norm = current_segment['Volume'] / current_segment['Volume'].mean()
    historical_vol_norm = historical_segment['Volume'] / historical_segment['Volume'].mean()
    
    fig.add_trace(
        go.Bar(
            x=current_dates,
            y=current_vol_norm,
            name='Current Vol',
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=historical_dates,
            y=historical_vol_norm,
            name='Historical Vol',
            marker_color='rgba(255, 0, 0, 0.5)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Pattern Comparison",
        xaxis_title="Trading Days",
        yaxis_title="Normalized Price (Start=100)",
        xaxis2_title="Trading Days",
        yaxis2_title="Normalized Volume",
        legend_title="Pattern Type",
        height=600
    )
    
    return fig

def visualize_future_trajectories(data, trajectories, avg_trajectory, forecast_horizon=30):
    """Visualize potential future trajectories based on historical resonances"""
    fig = go.Figure()
    
    # Plot historical data (last 60 days)
    historical_days = 60
    historical_data = data['Close'].iloc[-historical_days:]
    
    fig.add_trace(go.Scatter(
        x=list(range(-historical_days+1, 1)),
        y=historical_data.values,
        mode='lines',
        name='Historical',
        line=dict(color='black', width=2)
    ))
    
    # Plot individual trajectories with transparency based on resonance score
    for i, traj in enumerate(trajectories):
        score = traj['resonance_score']
        opacity = 0.3 + min(0.7, score)  # Scale opacity between 0.3 and 1.0
        
        fig.add_trace(go.Scatter(
            x=list(range(1, forecast_horizon+1)),
            y=traj['trajectory'],
            mode='lines',
            name=f"Trajectory {i+1}",
            line=dict(color=f'rgba(100,100,255,{opacity})'),
            hovertext=f"Start Date: {traj['start_date'].strftime('%Y-%m-%d')}<br>Score: {score:.4f}"
        ))
    
    # Plot average trajectory
    if len(avg_trajectory) > 0:
        fig.add_trace(go.Scatter(
            x=list(range(1, forecast_horizon+1)),
            y=avg_trajectory,
            mode='lines',
            name='Average Trajectory',
            line=dict(color='red', width=3)
        ))
    
    # Add a vertical line at current time
    fig.add_shape(
        type="line",
        x0=0, y0=min(historical_data.min(), min([t['trajectory'].min() for t in trajectories]) if trajectories else historical_data.min()),
        x1=0, y1=max(historical_data.max(), max([t['trajectory'].max() for t in trajectories]) if trajectories else historical_data.max()),
        line=dict(color="black", width=2, dash="dash")
    )
    
    fig.update_layout(
        title="Potential Future Trajectories Based on Resonance Patterns",
        xaxis_title="Days (Negative=Past, Positive=Future)",
        yaxis_title="Price",
        hovermode="closest",
        height=500
    )
    
    return fig

def create_temporal_harmony_dashboard(data, resonances, window_size):
    """Create visualizations showing resonance strength across different timeframes"""
    if resonances is None or resonances.empty:
        return None
    
    # Create datetime bins for visualization
    resonances['Year'] = resonances['Start_Date'].dt.year
    resonances['Month'] = resonances['Start_Date'].dt.month
    resonances['Quarter'] = resonances['Start_Date'].dt.quarter
    
    # Create heatmap data for years and months
    year_counts = resonances.groupby('Year')['Resonance_Score'].mean().reset_index()
    month_counts = resonances.groupby('Month')['Resonance_Score'].mean().reset_index()
    
    # Year resonance
    year_fig = px.bar(
        year_counts,
        x='Year',
        y='Resonance_Score',
        title='Resonance Strength by Year',
        labels={'Resonance_Score': 'Average Resonance'},
        color='Resonance_Score',
        color_continuous_scale='Viridis'
    )
    
    # Month resonance (cyclic pattern)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_counts['Month_Name'] = month_counts['Month'].apply(lambda x: month_names[x-1])
    
    month_fig = px.bar(
        month_counts,
        x='Month',
        y='Resonance_Score',
        title='Seasonal Resonance Strength',
        labels={'Resonance_Score': 'Average Resonance', 'Month': 'Month'},
        color='Resonance_Score',
        color_continuous_scale='Viridis'
    )
    month_fig.update_layout(xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(1, 13)),
        ticktext = month_names
    ))
    
    # Create timeseries of all resonances
    timeseries_fig = px.scatter(
        resonances,
        x='Start_Date',
        y='Resonance_Score',
        color='Resonance_Score',
        size='Resonance_Score',
        title='Historical Resonance Distribution',
        labels={'Resonance_Score': 'Resonance Strength', 'Start_Date': 'Date'},
        color_continuous_scale='Viridis'
    )
    timeseries_fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    return year_fig, month_fig, timeseries_fig

def calculate_fractal_dimension(data, window_sizes):
    """Calculate fractal dimension using box counting method"""
    dimensions = []
    
    normalized_data = normalize_series(data)
    
    for size in window_sizes:
        # Count number of boxes needed to cover the data
        n_boxes = len(data) // size
        count = 0
        
        for i in range(n_boxes):
            start_idx = i * size
            end_idx = start_idx + size
            
            if end_idx <= len(data):
                # Check if this box contains part of the curve
                segment = normalized_data[start_idx:end_idx]
                if segment.max() - segment.min() > 0:
                    count += 1
        
        dimensions.append((np.log(1/size), np.log(count)))
    
    # Calculate slope of log-log plot
    if dimensions:
        x, y = zip(*dimensions)
        x, y = np.array(x), np.array(y)
        
        # Remove any inf or NaN values
        valid_indices = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        x, y = x[valid_indices], y[valid_indices]
        
        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return slope
    
    return None

def calculate_cycle_periods(data, column='Close'):
    """Calculate dominant cycle periods using FFT"""
    # Detrend data
    price = data[column].values
    x = np.arange(len(price))
    
    # Fit linear trend
    slope, intercept = np.polyfit(x, price, 1)
    trend = slope * x + intercept
    
    # Remove trend
    detrended = price - trend
    
    # Apply FFT
    fft_result = np.fft.fft(detrended)
    frequency = np.fft.fftfreq(len(detrended))
    
    # Get positive frequencies only
    positive_freq_idx = np.where(frequency > 0)
    frequency = frequency[positive_freq_idx]
    magnitude = np.abs(fft_result[positive_freq_idx])
    
    # Find peaks in the frequency domain
    peaks, _ = signal.find_peaks(magnitude, height=np.mean(magnitude))
    
    # Calculate period from frequency
    if len(peaks) > 0:
        peak_freqs = frequency[peaks]
        peak_magnitudes = magnitude[peaks]
        
        # Convert frequency to periods (in days)
        periods = np.round(1 / peak_freqs).astype(int)
        
        # Sort by magnitude (importance)
        sorted_idx = np.argsort(peak_magnitudes)[::-1]
        sorted_periods = periods[sorted_idx]
        sorted_magnitudes = peak_magnitudes[sorted_idx]
        
        # Return top periods and their strengths
        return sorted_periods[:5], sorted_magnitudes[:5] / np.sum(peak_magnitudes)
    
    return [], []

# ------------------- Application Layout -------------------

def main():
    st.title("Financial Time Resonance Engine")
    st.markdown("""
    This application identifies temporal resonance patterns in financial markets - when current market 
    conditions share harmonic relationships with multiple historical periods across different dimensions of time.
    """)

    # Sidebar for controls
    st.sidebar.header("Analysis Controls")
    
    uploaded_file = st.sidebar.file_uploader("Upload financial data (CSV)", type=["csv"])
    
    if not uploaded_file:
        st.info("Please upload a CSV file with columns: Date, Open, High, Low, Close, Adj Close, Volume")
        st.stop()
    
    # Load data
    data = load_data(uploaded_file)
    
    # Display basic statistics about the data
    st.sidebar.subheader("Data Statistics")
    st.sidebar.text(f"Date Range: {data.index[0].date()} to {data.index[-1].date()}")
    st.sidebar.text(f"Total Trading Days: {len(data)}")
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    window_size = st.sidebar.slider("Analysis Window Size (days)", 5, 252, 60)
    n_results = st.sidebar.slider("Number of Resonance Patterns", 3, 10, 5)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 5, 90, 30)
    
    # Feature importance weights
    st.sidebar.subheader("Feature Importance")
    price_weight = st.sidebar.slider("Price Pattern Importance", 0.1, 1.0, 0.6)
    volume_weight = st.sidebar.slider("Volume Pattern Importance", 0.0, 1.0, 0.2)
    volatility_weight = st.sidebar.slider("Volatility Pattern Importance", 0.0, 1.0, 0.2)
    
    # Main analysis
    # Ensure we have enough data for the analysis
    if len(data) < window_size * 2:
        st.error(f"Not enough data for analysis. Need at least {window_size * 2} days, but got {len(data)}.")
        st.stop()
    
    # Get current window
    current_window = data.iloc[-window_size:]
    
    # Find resonance patterns
    with st.spinner("Finding resonance patterns..."):
        resonances, historical_segments = find_resonances(data, current_window, window_size, n_results)
    
    # Main display area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resonance Dashboard", 
        "Pattern Comparison", 
        "Future Trajectories",
        "Temporal Harmony",
        "Fractal Analysis"
    ])
    
    with tab1:
        st.header("Market Resonance Dashboard")
        
        # Current market status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Market Status")
            last_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1
            
            st.metric(
                label="Last Price", 
                value=f"{last_price:.2f}",
                delta=f"{price_change:.2%}"
            )
            
            # Recent trend visualization
            recent_data = data.iloc[-60:]
            fig = px.line(
                recent_data, 
                y='Close', 
                title='Recent Price Trend (60 Days)',
                labels={'value': 'Price', 'index': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Resonance Patterns")
            
            if resonances is not None and not resonances.empty:
                # Create a formatted table of top resonances
                for i, (idx, row) in enumerate(resonances.iterrows()):
                    expander = st.expander(f"Pattern {i+1}: {row['Start_Date'].strftime('%Y-%m-%d')} (Score: {row['Resonance_Score']:.4f})")
                    
                    with expander:
                        # Get the actual data segment
                        segment = historical_segments[i]
                        
                        # Display segment date range
                        st.text(f"Period: {segment.index[0].strftime('%Y-%m-%d')} to {segment.index[-1].strftime('%Y-%m-%d')}")
                        
                        # Create a mini chart of this pattern
                        segment_fig = px.line(segment, y='Close', title=f"Price Pattern")
                        segment_fig.update_layout(height=200, margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(segment_fig, use_container_width=True)
                        
                        # Calculate what happened after this pattern
                        start_idx = data.index.get_indexer([segment.index[0]])[0]
                        
                        if start_idx + window_size + 30 <= len(data):
                            future_segment = data.iloc[start_idx+window_size:start_idx+window_size+30]
                            future_return = future_segment['Close'].iloc[-1] / segment['Close'].iloc[-1] - 1
                            
                            st.metric(
                                label="30-Day Forward Return", 
                                value=f"{future_return:.2%}",
                                delta=f"{future_return:.2%}"
                            )
            else:
                st.warning("No significant resonance patterns found.")
        
        # 3D Resonance Field Map
        if resonances is not None and not resonances.empty:
            st.subheader("3D Resonance Field Map")
            
            with st.spinner("Generating 3D Resonance Map..."):
                fig_3d = visualize_resonances_3d(
                    current_window, 
                    historical_segments, 
                    resonances['Resonance_Score'].values
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.caption("""
                This 3D map shows the relationship between the current market position (red diamond) 
                and historical resonance patterns. Closer points have higher pattern similarity.
                """)
    
    with tab2:
        st.header("Pattern Comparison Analysis")
        
        if resonances is not None and not resonances.empty:
            # Select pattern to compare
            pattern_idx = st.selectbox(
                "Select pattern to compare with current market",
                range(len(resonances)),
                format_func=lambda i: f"Pattern {i+1}: {resonances['Start_Date'].iloc[i].strftime('%Y-%m-%d')} (Score: {resonances['Resonance_Score'].iloc[i]:.4f})"
            )
            
            # Get the selected historical segment
            historical_segment = historical_segments[pattern_idx]
            
            # Create comparison visualization
            with st.spinner("Generating comparison visualization..."):
                fig = compare_patterns(current_window, historical_segment)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation statistics
                price_corr, _ = pearsonr(
                    normalize_series(current_window['Close']),
                    normalize_series(historical_segment['Close'])
                )
                volume_corr, _ = pearsonr(
                    normalize_series(current_window['Volume']),
                    normalize_series(historical_segment['Volume'])
                )
                
                # Display correlation metrics
                col1, col2 = st.columns(2)
                col1.metric("Price Pattern Correlation", f"{price_corr:.4f}")
                col2.metric("Volume Pattern Correlation", f"{volume_corr:.4f}")
                
                # What happened next
                st.subheader("What Happened Next")
                
                # Get index of this historical segment
                start_idx = data.index.get_indexer([historical_segment.index[0]])[0]
                
                if start_idx + window_size + forecast_horizon <= len(data):
                    # Extract the future trajectory after this segment
                    future_segment = data.iloc[start_idx+window_size:start_idx+window_size+forecast_horizon]
                    
                    # Create visualization of what happened after this pattern
                    fig = px.line(
                        future_segment,
                        y='Close',
                        title=f"Market Movement After Selected Pattern ({forecast_horizon} Days)",
                        labels={'value': 'Price', 'index': 'Date'}
                    )
                    
                    # Calculate key metrics
                    future_return = future_segment['Close'].iloc[-1] / historical_segment['Close'].iloc[-1] - 1
                    max_drawdown = (future_segment['Close'].min() / historical_segment['Close'].iloc[-1]) - 1
                    max_upside = (future_segment['Close'].max() / historical_segment['Close'].iloc[-1]) - 1
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{forecast_horizon}-Day Return", f"{future_return:.2%}")
                    col2.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
                    col3.metric("Maximum Upside", f"{max_upside:.2%}")
                else:
                    st.info("Insufficient data to show future movement after this pattern.")
        else:
            st.warning("No significant resonance patterns found for comparison.")
    
    with tab3:
        st.header("Future Trajectories Analysis")
        if resonances is not None and not resonances.empty:
            with st.spinner("Calculating potential future trajectories..."):
                trajectories, avg_trajectory = calculate_future_trajectories(
                    data, resonances, historical_segments, window_size, forecast_horizon
                )

                # Filter out invalid or empty trajectories
                valid_trajectories = [
                    t for t in trajectories if isinstance(t.get('trajectory', None), (list, np.ndarray)) and len(t['trajectory']) > 0
                ]

                if valid_trajectories:
                    # Visualize trajectories
                    fig = visualize_future_trajectories(
                        data, valid_trajectories, avg_trajectory, forecast_horizon
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate probability distribution of outcomes
                    end_values = [t['trajectory'][-1] for t in valid_trajectories]
                    baseline = data['Close'].iloc[-1]
                    returns = [(val / baseline - 1) for val in end_values]

                    # Create histogram of potential returns
                    hist_fig = px.histogram(
                        x=returns,
                        nbins=10,
                        title=f"Distribution of Potential {forecast_horizon}-Day Returns",
                        labels={'x': 'Return', 'y': 'Frequency'},
                        color_discrete_sequence=['lightblue']
                    )

                    # Add vertical lines for mean and zero return
                    mean_return = np.mean(returns)
                    hist_fig.add_vline(x=mean_return, line_dash="dash", line_color="red")
                    hist_fig.add_vline(x=0, line_dash="solid", line_color="black")

                    # Determine y-position for annotation
                    y_values = []
                    for trace in hist_fig.data:
                        if hasattr(trace, 'y') and trace.y is not None:
                            y_values.extend(trace.y)
                    y_max = max(y_values) if y_values else 0

                    # Annotate mean return
                    hist_fig.add_annotation(
                        x=mean_return,
                        y=y_max * 0.9,
                        text=f"Mean: {mean_return:.2%}",
                        showarrow=True,
                        arrowhead=1
                    )

                    st.plotly_chart(hist_fig, use_container_width=True)

                    # Display return probabilities
                    st.subheader("Return Probabilities")
                    prob_positive = sum(1 for r in returns if r > 0) / len(returns)
                    prob_negative = 1 - prob_positive
                    prob_very_positive = sum(1 for r in returns if r > 0.05) / len(returns)
                    prob_very_negative = sum(1 for r in returns if r < -0.05) / len(returns)

                    col1, col2 = st.columns(2)
                    col1.metric("Probability of Positive Return", f"{prob_positive:.1%}")
                    col2.metric("Probability of Negative Return", f"{prob_negative:.1%}")

                    col3, col4 = st.columns(2)
                    col3.metric("Probability of >5% Return", f"{prob_very_positive:.1%}")
                    col4.metric("Probability of <-5% Return", f"{prob_very_negative:.1%}")
                else:
                    st.warning("Insufficient historical data to project future trajectories.")
        else:
            st.warning("No significant resonance patterns found for trajectory analysis.")

    
    with tab4:
        st.header("Temporal Harmony Analysis")
        
        if resonances is not None and not resonances.empty:
            # Create temporal harmony visualizations
            year_fig, month_fig, timeseries_fig = create_temporal_harmony_dashboard(
                data, resonances, window_size
            )
            
            # Display visualizations
            st.plotly_chart(year_fig, use_container_width=True)
            st.plotly_chart(month_fig, use_container_width=True)
            st.plotly_chart(timeseries_fig, use_container_width=True)
            
            # Calculate dominant cycle periods
            with st.spinner("Calculating market cycles..."):
                periods, strengths = calculate_cycle_periods(data)
                
                if len(periods) > 0:
                    st.subheader("Dominant Market Cycles")
                    
                    # Create a bar chart of dominant cycles
                    cycle_data = pd.DataFrame({
                        'Period (Days)': periods,
                        'Strength': strengths
                    })
                    
                    fig = px.bar(
                        cycle_data,
                        x='Period (Days)',
                        y='Strength',
                        title='Dominant Market Cycles',
                        labels={'Strength': 'Relative Importance'},
                        color='Strength',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the cycle data
                    st.write("Dominant Cycle Periods:")
                    
                    for i, (period, strength) in enumerate(zip(periods, strengths)):
                        if i < 5:  # Show top 5 cycles
                            st.write(f"- {period} days (Strength: {strength:.2f})")
                else:
                    st.info("No significant cycles detected in the data.")
        else:
            st.warning("No significant resonance patterns found for temporal harmony analysis.")
    
    with tab5:
        st.header("Fractal Analysis")
        
        # Calculate fractal dimension
        with st.spinner("Calculating fractal dimension..."):
            window_sizes = [5, 10, 20, 40, 80]
            fractal_dim = calculate_fractal_dimension(data['Close'], window_sizes)
            
            if fractal_dim is not None:
                st.subheader("Fractal Dimension Analysis")
                
                col1, col2 = st.columns(2)
                col1.metric("Fractal Dimension", f"{fractal_dim:.4f}")
                
                # Interpretation
                if fractal_dim < 1.3:
                    col2.info("Low complexity: Market shows trend-following behavior")
                elif fractal_dim < 1.6:
                    col2.info("Moderate complexity: Market shows mixed behavior")
                else:
                    col2.info("High complexity: Market shows chaotic behavior")
            else:
                st.warning("Unable to calculate fractal dimension from the given data.")
        
        # Multi-scale wavelet decomposition
        st.subheader("Multi-scale Wavelet Decomposition")
        
        # Perform wavelet decomposition
        price_data = data['Close'].values
        wavelet = 'db4'
        level = 4
        
        coeffs = pywt.wavedec(price_data, wavelet, level=level)
        
        # Plot the wavelet decomposition
        fig, axes = plt.subplots(level + 1, 1, figsize=(10, 10))
        
        # Approximation coefficients
        axes[0].plot(coeffs[0])
        axes[0].set_title('Approximation (Long-term trend)')
        axes[0].set_xlim(0, len(coeffs[0]))
        
        # Detail coefficients
        for i in range(1, level + 1):
            axes[i].plot(coeffs[i])
            axes[i].set_title(f'Detail Level {i} (Period: ~{2**i} days)')
            axes[i].set_xlim(0, len(coeffs[i]))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Wavelet Decomposition Interpretation:**
        - **Approximation (top)**: Shows the long-term trend component
        - **Detail Level 1**: Shows the highest frequency components (daily fluctuations)
        - **Detail Level 2-4**: Show progressively lower frequency components (weekly to monthly patterns)
        
        This decomposition helps identify which time scales contribute most to the current market behavior.
        """)
        
        # Self-similarity analysis
        st.subheader("Self-similarity Analysis")
        
        # Create a heatmap of self-similarity at different scales
        with st.spinner("Calculating self-similarity matrix..."):
            n_segments = 10
            segment_size = len(data) // n_segments
            
            similarity_matrix = np.zeros((n_segments, n_segments))
            
            for i in range(n_segments):
                for j in range(n_segments):
                    segment_i = data['Close'].iloc[i*segment_size:(i+1)*segment_size].values
                    segment_j = data['Close'].iloc[j*segment_size:(j+1)*segment_size].values
                    
                    # Normalize segments
                    segment_i = (segment_i - segment_i.min()) / (segment_i.max() - segment_i.min())
                    segment_j = (segment_j - segment_j.min()) / (segment_j.max() - segment_j.min())
                    
                    # Calculate correlation
                    corr, _ = pearsonr(segment_i, segment_j)
                    similarity_matrix[i, j] = abs(corr)
            
            # Create heatmap
            fig = px.imshow(
                similarity_matrix,
                labels=dict(x="Segment", y="Segment", color="Correlation"),
                x=[f"{i+1}" for i in range(n_segments)],
                y=[f"{i+1}" for i in range(n_segments)],
                title="Self-similarity Matrix (Pattern Recurrence)",
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Self-similarity Matrix Interpretation:**
            - Bright diagonal: Each segment is perfectly correlated with itself
            - Bright off-diagonal areas: Time periods with similar pattern structure
            - Dark areas: Time periods with dissimilar patterns
            
            This visualization helps identify which historical periods share similar pattern structures.
            """)

    # Bottom section for insights and alerts
    st.header("Market Insights and Alerts")
    
    # Generate insights based on the analysis
    if resonances is not None and not resonances.empty:
        # Calculate average future return based on resonance patterns
        future_returns = []
        for i, segment in enumerate(historical_segments):
            start_idx = data.index.get_indexer([segment.index[0]])[0]
            
            if start_idx + window_size + forecast_horizon <= len(data):
                future_segment = data.iloc[start_idx+window_size:start_idx+window_size+forecast_horizon]
                future_return = future_segment['Close'].iloc[-1] / segment['Close'].iloc[-1] - 1
                future_returns.append((future_return, resonances['Resonance_Score'].iloc[i]))
        
        if future_returns:
            # Calculate weighted average return
            weighted_sum = sum(ret * score for ret, score in future_returns)
            total_weight = sum(score for _, score in future_returns)
            avg_expected_return = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Display insights
            if avg_expected_return > 0.05:
                st.success(f"⚠️ Strong bullish resonance detected! Average historical return: {avg_expected_return:.2%}")
            elif avg_expected_return > 0.02:
                st.info(f"📈 Moderate bullish resonance detected. Average historical return: {avg_expected_return:.2%}")
            elif avg_expected_return < -0.05:
                st.error(f"⚠️ Strong bearish resonance detected! Average historical return: {avg_expected_return:.2%}")
            elif avg_expected_return < -0.02:
                st.warning(f"📉 Moderate bearish resonance detected. Average historical return: {avg_expected_return:.2%}")
            else:
                st.info(f"📊 Neutral market resonance detected. Average historical return: {avg_expected_return:.2%}")
        
        # Check for extreme resonance scores
        max_resonance = resonances['Resonance_Score'].max()
        if max_resonance > 0.8:
            st.warning(f"⚠️ Unusually strong historical pattern match detected (Score: {max_resonance:.4f})")
    
    # Add an export button
    if resonances is not None and not resonances.empty:
        st.subheader("Export Analysis Results")
        
        # Create a simple report of the analysis
        buffer = io.StringIO()
        
        buffer.write("# Financial Time Resonance Analysis Report\n\n")
        buffer.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        buffer.write(f"Data Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n")
        buffer.write(f"Window Size: {window_size} days\n")
        buffer.write(f"Forecast Horizon: {forecast_horizon} days\n\n")
        
        buffer.write("## Top Resonance Patterns\n\n")
        for i, (idx, row) in enumerate(resonances.iterrows()):
            buffer.write(f"Pattern {i+1}: {row['Start_Date'].strftime('%Y-%m-%d')} (Score: {row['Resonance_Score']:.4f})\n")
        
        buffer.write("\n## Market Outlook\n\n")
        if 'avg_expected_return' in locals():
            buffer.write(f"Expected {forecast_horizon}-Day Return: {avg_expected_return:.2%}\n")
        
        # Convert buffer to bytes for download
        report_content = buffer.getvalue()
        
        # Create download button
        st.download_button(
            label="Download Analysis Report",
            data=report_content,
            file_name="resonance_analysis_report.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
