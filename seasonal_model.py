# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:58:53 2025

@author: d23gr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Generate synthetic seasonal data (for demonstration)
np.random.seed(42)
time = np.arange(100)
seasonal_effect = 10 * np.sin(time * (2 * np.pi / 12))  # 12-period seasonality
trend = time * 0.1  # Slight upward trend
noise = np.random.normal(scale=2, size=100)  # Random noise
data = seasonal_effect + trend + noise

# Create a pandas DataFrame
df = pd.DataFrame({"Date": pd.date_range(start="2020-01-01", periods=100, freq="M"), "Value": data})
df.set_index("Date", inplace=True)

# Plot the original time series
df.plot(figsize=(10, 5))
plt.title("Synthetic Seasonal Time Series Data")
plt.show()

# Check stationarity
result = adfuller(df["Value"])
print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")

# Decompose time series
decomposed = seasonal_decompose(df["Value"], model="additive", period=12)
decomposed.plot()
plt.show()

# Fit SARIMA model (SARIMA(p,d,q)(P,D,Q,s))
model = SARIMAX(df["Value"], order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
sarima_result = model.fit()

# Forecast next 12 months
forecast = sarima_result.get_forecast(steps=12)
forecast_index = pd.date_range(start=df.index[-1], periods=12, freq="M")
forecast_mean = forecast.predicted_mean

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Value"], label="Observed")
plt.plot(forecast_index, forecast_mean, label="Forecast", color="red")
plt.legend()
plt.title("SARIMA Model Forecast")
plt.show()

# Great question! Stationarity is a key concept in time series analysis because many forecasting models, like ARIMA and SARIMA, assume the data is stationary—meaning its statistical properties (mean, variance, covariance) remain constant over time.

### **Stationarity Check: Augmented Dickey-Fuller (ADF) Test**
# The **ADF test** helps determine whether a time series is stationary by checking if it has a unit root (which indicates non-stationarity). The hypothesis setup is:
# - **Null hypothesis (H₀)**: The time series has a unit root (i.e., it is non-stationary).
# - **Alternative hypothesis (H₁)**: The time series is stationary.

# The test produces an **ADF statistic** and a **p-value**:
# - If the **p-value < 0.05**, we reject the null hypothesis—suggesting the data is stationary.
# - If the **p-value ≥ 0.05**, we fail to reject the null hypothesis—suggesting the data is non-stationary.

### **Next Steps Based on Results**
#### **1. If the result is significant (p-value < 0.05, meaning the data is stationary)**
# ✅ Proceed with modeling (e.g., ARIMA, SARIMA) without further transformations.

# #### **2. If the result is not significant (p-value ≥ 0.05, meaning the data is non-stationary)**
# 🔹 **Differencing** – One of the most common techniques:
#    - Compute the difference between consecutive observations:  
#      \( y_t' = y_t - y_{t-1} \)
#    - If the first difference doesn’t stabilize the mean/variance, try second-order differencing.

# 🔹 **Transformations** – If the variance isn't stable:
#    - Log transformation: \( y_t' = \log(y_t) \)
#    - Square root or Box-Cox transformation for more severe variance issues.

# 🔹 **Seasonal differencing** – If periodic fluctuations persist:
#    - Compute differences between values at fixed seasonal lags (e.g., for monthly data, subtract previous year's value).

# After applying these transformations, **rerun the ADF test** to confirm stationarity. Once stationary, you can proceed with modeling!

### Decomposition
# The **seasonal decomposition** function is a powerful tool that breaks down a time series into three key components:  

# ### **1. Trend**
#    - The long-term movement in the data.  
#    - If the values increase or decrease over time, this component captures that pattern.  

# ### **2. Seasonality**
#    - The repeating cycles or periodic fluctuations in the data.  
#    - For example, retail sales often spike in December due to holiday shopping—this pattern repeats annually.  

# ### **3. Residual (Noise)**
#    - The random variations that don’t follow a clear trend or seasonal pattern.  
#    - It represents unpredictable factors affecting the data.  

# #### **How the Decomposition Works**
# - The function **estimates** each of these components using statistical techniques.  
# - If using **additive decomposition**, the time series is modeled as:  
#   \[
#   Y_t = Trend_t + Seasonality_t + Residual_t
#   \]
# - If using **multiplicative decomposition** (for data where seasonal effects grow over time), the model is:  
#   \[
#   Y_t = Trend_t \times Seasonality_t \times Residual_t
#   \]
# - The decomposition allows us to analyze each component separately, helping with forecasting and anomaly detection.  

# #### **Why This is Useful**
# - Identifies seasonality patterns for better predictions.  
# - Helps decide if differencing is needed for stationarity.  
# - Improves time series models like ARIMA or SARIMA by removing unwanted noise.  

# The decomposition process separates a time series into **trend** and **seasonality** using different smoothing and filtering techniques.

### **Trend Extraction**
# - The **trend component** captures the long-term direction of the data.
# - Common methods to determine the trend include:
#   - **Moving Average Smoothing**: Calculates the average of nearby points (e.g., 12-month moving average for annual trends).
#   - **Polynomial Regression**: Fits a smooth polynomial curve to detect gradual changes.
#   - **Loess (Locally Weighted Regression)**: A flexible non-parametric approach that adjusts to local fluctuations.

# ### **Seasonality Extraction**
# - The **seasonal component** isolates repeating patterns at regular intervals.
# - Methods used:
#   - **Moving Average Subtraction**: Removes the trend to expose seasonality.
#   - **Fourier Transforms**: Analyzes cyclic frequencies in the data.
#   - **Seasonal Period Detection**: Uses periodicity assumptions (e.g., monthly cycles in yearly data) to extract repeating behavior.

# ### **How the Algorithm Works**
# 1. Estimate the **trend** using moving averages or polynomial smoothing.
# 2. Remove the trend from the original data to get **detrended values**.
# 3. Identify and extract the **seasonal pattern** from the detrended series.
# 4. Compute the **residuals** by subtracting both trend and seasonality from the original data.

# Both **smoothing** and **linear regression** are used to identify trends in time series data, but they have distinct strengths and weaknesses.

# ### **Smoothing vs. Linear Regression for Trend Detection**
# #### **Smoothing**
# **Definition:** Smoothing applies techniques like moving averages or Loess regression to reduce short-term fluctuations and highlight longer-term trends.

# ✅ **Advantages:**
# - **Handles Nonlinear Trends:** Works well when trends are not strictly linear, such as curved or cyclical patterns.
# - **Preserves Local Structure:** Adapts to varying trends over time rather than enforcing a global trend.
# - **Reduces Noise:** Filters out short-term irregularities to reveal underlying behavior.

# ❌ **Disadvantages:**
# - **Less Predictive Power:** Primarily descriptive; does not provide an explicit mathematical formula for future forecasts.
# - **May Oversmooth Data:** Can obscure important changes if the smoothing window is too large.

# #### **Linear Regression**
# **Definition:** Linear regression fits a straight line to the data by minimizing the difference between observed and predicted values.

# ✅ **Advantages:**
# - **Provides a Clear Model:** Generates an equation for predicting future values.
# - **Mathematically Interpretable:** Coefficients explain the rate of change over time.
# - **Works Well for Simple Trends:** Ideal when data follows a steady linear pattern.

# ❌ **Disadvantages:**
# - **Limited to Linear Relationships:** Fails to capture nonlinear trends.
# - **Sensitive to Outliers:** Extreme values can skew the trend significantly.
# - **Ignores Seasonality:** Cannot account for periodic fluctuations on its own.

# ### **Why Smoothing is Preferred for Trend Extraction?**
# - Time series data often contains **nonlinear and seasonal patterns** that simple linear regression cannot capture.
# - Smoothing techniques (like moving averages) can adapt to changes **without enforcing a fixed trend model**.
# - Linear regression assumes **a single trend structure**, while smoothing allows trends to evolve naturally.

# #### **When to Use Each?**
# - **Use smoothing** when detecting underlying trends without enforcing a strict model.
# - **Use linear regression** when a simple, **predictive model** is required.

# In fact, using **both a linear model for trend** and **Fourier analysis for seasonality** is a well-established approach in time series modeling. However, there are practical challenges and trade-offs when combining them.

# ### **Why This Approach Can Work**
# 1. **Linear Trend:** A regression model can estimate a steady increasing or decreasing pattern in the data.
# 2. **Fourier Analysis:** Uses sine and cosine waves to model repeating seasonal components.
# 3. **Combined Model:** The total signal can be represented as:
#    \[
#    Y_t = \beta_0 + \beta_1 t + \sum_{k=1}^{K} (\alpha_k \cos(2\pi kt / T) + \gamma_k \sin(2\pi kt / T)) + \varepsilon_t
#    \]
#    where:
#    - \( \beta_0 + \beta_1 t \) represents the linear trend,
#    - The summation term represents the seasonal cycles extracted from Fourier analysis.

# ### **Challenges of Combining Both**
# 1. **Nonlinearity in Trends:** Many real-world trends are nonlinear (e.g., logistic growth, exponential decay), and a linear model might oversimplify complex changes.
# 2. **Seasonality Complexity:** Fourier series assumes **perfect periodic cycles**, but real-world seasonality often has slight variations that can't be captured with simple sinusoidal terms.
# 3. **Data Length Dependency:** Fourier methods rely on adequate historical data to estimate stable frequencies—short time series may not provide meaningful frequency information.
# 4. **Noise Sensitivity:** If the time series contains irregular variations or external factors (e.g., economic shocks), Fourier analysis might overfit or struggle to distinguish real seasonality from noise.

# ### **Alternative Approach**
# A powerful alternative is using a **state-space model** (like an exponential smoothing model or Kalman filter) that dynamically updates trends and seasonality while accounting for uncertainty.

# That being said, in certain applications like **financial modeling or climate data**, blending **regression for trends** and **Fourier basis functions for seasonality** is a popular technique! Would you like an example of implementing this combination in Python?

## LINEAR MODEL PLUS FOURIER ANALYSIS ON ENERGY GRID DATA
# Fourier Analysis + Linear Trend for Electric Grid Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from sklearn.linear_model import LinearRegression

# Load synthetic regional electric grid data (replace with actual dataset)
np.random.seed(42)
months = np.arange(120)  # 10 years of monthly data
trend = months * 0.5  # Simulated linear trend
seasonal_effect = 20 * np.sin(2 * np.pi * months / 12)  # Annual seasonality
noise = np.random.normal(scale=5, size=120)  # Random noise
electric_demand = trend + seasonal_effect + noise

# Create DataFrame
df = pd.DataFrame({"Month": months, "Demand": electric_demand})

# Apply Linear Regression for Trend Extraction
X = df["Month"].values.reshape(-1, 1)
y = df["Demand"].values
model = LinearRegression()
model.fit(X, y)
trend_estimate = model.predict(X)

# Fourier Transform for Seasonality Extraction
fft_values = fft(y - trend_estimate)  # Remove trend before applying FFT
frequencies = fftfreq(len(y), d=1)  # Monthly intervals

# Plot Results
plt.figure(figsize=(12, 5))
plt.plot(df["Month"], df["Demand"], label="Observed Demand", alpha=0.6)
plt.plot(df["Month"], trend_estimate, label="Linear Trend", linestyle="dashed", color="red")
plt.legend()
plt.title("Electric Grid Demand with Linear Trend")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(frequencies[:60], np.abs(fft_values[:60]), label="Fourier Spectrum")
plt.title("Fourier Analysis of Seasonal Patterns")
plt.xlabel("Frequency (cycles per month)")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

