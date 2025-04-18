
# 🕰️ Financial Time Resonance Engine

A powerful Streamlit application that uncovers **temporal resonance patterns** in financial market data. It helps identify historical windows that harmonically align with current market behavior across time, price, volume, and volatility dimensions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://timemachine.streamlit.app/)

---

## 🚀 Features

- 📊 Interactive dashboard with multi-tab layout
- 🔍 Upload custom financial datasets in CSV format
- ⏱️ Detect harmonic relationships with historical price/volume/volatility windows
- 🔄 Compare past and present market patterns using t-SNE, UMAP, and FastDTW
- 🔮 Forecast potential future market trajectories based on resonance similarity
- 🌐 Fully built using Streamlit for real-time, web-based interactivity

---

## 📁 How to Use (Locally)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/timemachine.git
cd timemachine
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📦 Requirements

Make sure you have Python **3.9+** installed. The key dependencies include:

- `streamlit`
- `pandas`
- `plotly`
- `altair`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `PyWavelets`
- `fastdtw`
- `umap-learn`
- `matplotlib`

Full list in [`requirements.txt`](./requirements.txt)

---

## 📈 Input Format

Upload a CSV file with the following columns:

```
Date, Open, High, Low, Close, Adj Close, Volume
```

Dates must be in chronological order. No missing or null values.

---

## 🧠 Behind the Scenes

The app performs:

- Rolling window analysis
- Volatility normalization
- Distance calculations using FastDTW & Euclidean metrics
- Dimensionality reduction (UMAP, t-SNE)
- Correlation & stationarity tests (ADF, Pearson)
- Wavelet transforms for fractal signal decomposition

---

## 💡 Inspired by

Temporal harmonics in market data analysis, signal processing, and fractal geometry in financial time series.

---

## 🌐 Hosted Version

Try it live:  
👉 [timemachine.streamlit.app](https://timemachine.streamlit.app/)

---

## 📬 Contact

Have questions or suggestions?  
Open an issue or reach out via [GitHub](https://github.com/sayantanmadal1).

---
