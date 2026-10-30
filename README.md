Weather History Forecasting (2006–2016)

Data Science & Analytics Course Project (IE333)

📌 Project Overview

Weather conditions—specifically temperature—have a profound impact on daily life and key industries such as agriculture, transportation, and energy. This project leverages over 10 years of historical hourly weather data to analyze patterns and build predictive time series models capable of forecasting daily temperatures.

By preprocessing raw weather data, engineering seasonal features, and implementing diverse forecasting methodologies, this project evaluates the strengths and trade-offs of statistical and machine learning approaches in weather prediction.

📊 Dataset Description

Source: Kaggle Weather Dataset

Timeframe: January 1, 2006 – December 31, 2016 (10+ years)

Original Granularity: Hourly (~96,000+ observations)

Resampled Granularity: Daily average

Key Target Variable: Temperature (C)

Features Included

Formatted Date: Timestamp of observation

Temperature (C): Target continuous variable ($\pm 8.8^\circ C$ std dev)

Humidity: Relative humidity

Wind Speed (km/h): Wind velocity

Pressure (millibars): Atmospheric pressure

TimeIndex: Sequential numeric counter representing time progression

sin / cos: Engineered 365-day Fourier features capturing annual seasonality

🛠️ Project Pipeline

Raw CSV Data ➔ Cleaning & Column Pruning ➔ Resampling to Daily ➔ Feature Engineering (Time Index + Fourier) ➔ Train/Test Split (80/20) ➔ Model Training & Evaluation


Data Cleaning: Removed noisy and non-predictive columns (Summary, Precip Type, Wind Bearing, Visibility, Loud Cover, Daily Summary, Apparent Temperature).

Datetime Formatting & Indexing: Converted dates to UTC datetime objects, set them as the DataFrame index, and sorted chronologically.

Resampling: Aggregated hourly logs to daily averages (resample('D').mean()).

Feature Engineering:

Created linear TimeIndex ($0, 1, 2, \dots, n-1$).

Created cyclical Fourier terms ($\sin$ and $\cos$) with a period of 365 days:

$$\text{sin} = \sin\left(\frac{2\pi \cdot \text{TimeIndex}}{365}\right), \quad \text{cos} = \cos\left(\frac{2\pi \cdot \text{TimeIndex}}{365}\right)$$

Data Splitting: Chronological 80/20 split to preserve temporal sequence and avoid data leakage (Train: 2006–2014, Test: 2014–2016).

📈 Modeling Strategies

We implemented and compared three distinct modeling techniques:

Baseline Linear Regression (Trend Only):

Formula: $\text{Temp}(t) = \beta_0 + \beta_1 \cdot \text{TimeIndex}$

Purpose: Establishes a simple linear baseline representing overall warming/cooling trends.

Fourier-Enhanced Regression (Trend + Seasonality):

Formula: $\text{Temp}(t) = \beta_0 + \beta_1 \cdot \text{TimeIndex} + \beta_2 \cdot \sin\left(\frac{2\pi t}{365}\right) + \beta_3 \cdot \cos\left(\frac{2\pi t}{365}\right)$

Purpose: Incorporates harmonic trigonometric terms to model periodic annual cycles explicitly.

ARIMA (5, 1, 1):

Components: Autoregressive ($p=5$), First-order differencing ($d=1$), Moving Average ($q=1$).

Purpose: Classical univariate time-series model capturing short-term autocorrelation and temporal dependencies.

🏆 Model Performance & Evaluation

Models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Coefficient of Determination ($R^2$).

1. Training Phase Performance (2006–2014)

Model

RMSE (°C)

MAPE (%)

$R^2$

Linear Regression

8.8730

5.4695%

0.0042

Fourier Regression

3.8227

1.0176%

0.8152

ARIMA(5,1,1)

2.1382

0.6639%

0.9422

2. Testing / Holdout Phase Performance (2014–2016)

Model

RMSE (°C)

MAPE (%)

$R^2$

Linear Regression

8.6908

6.1044%

-0.0844

ARIMA(5,1,1)

8.4933

5.7742%

-0.0356

Fourier Regression (Winner)

3.3341

0.8734%

0.8404

Key Takeaways & Findings

Fourier Regression was the best performing generalizable model, maintaining strong performance on unseen data ($R^2 = 0.8404$, $\text{RMSE} = 3.33^\circ C$). The explicit 365-day harmonic components captured the strong annual seasonality effectively.

ARIMA overfit the training dataset. While achieving $R^2 = 0.9422$ on training data, its lack of built-in seasonal terms caused significant performance degradation on holdout test data ($R^2 < 0$).

Linear Regression failed to capture the nonlinear, seasonal dynamics of temperature ($R^2 \approx 0$).

🚀 Key Recommendations & Future Work

SARIMA / SARIMAX Modeling: Implement Seasonal ARIMA to explicitly capture seasonal lag periods (e.g., $S=365$) and reduce holdout overfitting.

Multivariate Integration: Incorporate exogenous features (Humidity, Pressure, Wind Speed) as predictors using models like XGBoost, LightGBM, or Prophet.

Deep Learning Approaches: Explore Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) architectures for nonlinear, non-periodic temperature anomalies.

Interactive Dashboard: Build a streamlined web app (Streamlit/Gradio) allowing users to query future temperature forecasts for arbitrary future dates.

💻 Tech Stack & Dependencies

Language: Python 3.x

Environment: Google Colab

Key Libraries:

pandas, numpy: Data manipulation and preprocessing

matplotlib, seaborn: Exploratory data analysis and visualization

statsmodels: ARIMA modeling and seasonal decomposition

scikit-learn: Linear & Fourier regression modeling and metrics evaluation

joblib: Model serialization and persistence
