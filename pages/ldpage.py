import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.title("PPG Signal Processing")

def highpass_filter_iir(x, lowcut, dt):
    
    N = len(x)
    
    RC_hp = 1.0 / (2.0 * np.pi * lowcut)
    alpha_hp = RC_hp / (RC_hp + dt)
    
    baseline_removed = np.zeros(N)
    for i in range(1, N):
        baseline_removed[i] = alpha_hp * (baseline_removed[i-1] + x[i] - x[i-1])

    return baseline_removed

uploaded_file = st.file_uploader("📈", type=["csv"])

st.sidebar.header("Signal Parameters")
lowcut = st.sidebar.number_input("High-pass cutoff frequency (Hz)", value=0.7, step=0.05)
downsample_factor = st.sidebar.slider("Downsample factor", 1, 20, 4)

if uploaded_file:
    
    df = pd.read_csv(uploaded_file)
    
    df.columns = ['timestamp_raw', 'red']
    
    df['red'] = pd.to_numeric(df['red'], errors='coerce')
    
    df['timestamp_raw'] = pd.to_numeric(df['timestamp_raw'], errors='coerce')
    df['timestamp'] = df['timestamp_raw'] / 1000000.0
    
    min_t_abs = df['timestamp'].min()
    df['timestamp'] = df['timestamp'] - min_t_abs
    
    df = df.dropna(subset=['red', 'timestamp_raw']) 

    t = df['timestamp'].values
    x = df['red'].values
    N = len(x)
    
    st.write(f"**Loaded {N} samples**")
    st.write(f"**Time: {t[0]:.2f}s to {t[-1]:.2f}s** (Total duration: {(t[-1]-t[0]):.2f}s)")
    
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    st.write(f"**Sampling rate: {fs:.2f} Hz**")
    st.write("---")
    
    try:
        baseline_removed = highpass_filter_iir(x, lowcut, fs, dt)
    except Exception as e:
        st.error(f"Error saat filtering: {e}")
        baseline_removed = np.zeros(N)
    
    ds_t = t[::downsample_factor]
    ds_orig = x[::downsample_factor]
    ds_filt = baseline_removed[::downsample_factor]
    st.subheader("Original PPG Signal (Downsampled)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ds_t, y=ds_orig, mode='lines', line=dict(color='red')))
    fig1.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Baseline-Removed Signal")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ds_t, y=ds_filt, mode='lines', line=dict(color='green')))
    fig2.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Upload File PPG (.csv)")
