# Stock Market Analysis & Price Prediction using LSTM

A multivariate time-series project that analyses seven years of Google (GOOG) daily stock data and trains a deep LSTM network to predict the next day's closing price. The project covers the complete pipeline: data collection, exploratory analysis, preprocessing, model training, evaluation and an honest comparison against a simple baseline.

> **Disclaimer:** This is a learning project. It is not financial advice and should not be used to make trading decisions.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Phase 1: Exploratory Data Analysis](#5-phase-1-exploratory-data-analysis)
6. [Phase 2: Data Preprocessing](#6-phase-2-data-preprocessing)
7. [Phase 3: Model Training and Evaluation](#7-phase-3-model-training-and-evaluation)
8. [Results](#8-results)
9. [Interpreting the Graphs](#9-interpreting-the-graphs)
10. [Key Findings and Limitations](#10-key-findings-and-limitations)
11. [How to Run](#11-how-to-run)
12. [Future Work](#12-future-work)
13. [Acknowledgements](#13-acknowledgements)

---

## 1. Objective

The goal of this project is to build an end-to-end deep learning workflow for time-series forecasting and to evaluate it honestly.

Specifically, the project:

- collects historical GOOG stock data from Yahoo Finance,
- explores trends, volume behaviour and data quality,
- prepares the data for a sequence model without leaking future information,
- trains a 4-layer LSTM to predict the next day's closing price from the previous 60 trading days,
- measures performance on training, validation and test data, and
- compares the model against a naive baseline to check whether it has actually learned something useful.

---

## 2. Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.13 |
| Data collection | `yfinance` |
| Data handling | `pandas`, `numpy` |
| Visualisation | `matplotlib` |
| Preprocessing and metrics | `scikit-learn` (`MinMaxScaler`, error metrics), `joblib` |
| Deep learning | `tensorflow` / `keras` (TensorFlow 2.21) |
| Environment | `venv`, Jupyter notebooks in VS Code |
| Version control | Git and GitHub |

Training runs on the CPU. TensorFlow does not use the GPU on native Windows, and the dataset is small enough that this is not a problem (about 2 minutes 40 seconds for the full training run).

---

## 3. Project Structure

```
Stock-Price-Training/
├── data/
│   ├── raw_data/              # original data downloaded from Yahoo Finance
│   ├── cleaned_data/          # cleaned dataset produced in Phase 1
│   └── model_ready_data/      # train/val/test sequences (.npz) produced in Phase 2
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_data_preprocessing.ipynb
│   └── 3_models_training.ipynb
├── saved_models/
│   ├── lstm_model.keras       # trained LSTM (weights from the best epoch)
│   └── scaler.pkl             # MinMaxScaler fitted on the training data
├── results/
│   └── plots/                 # all generated figures
├── scripts/
│   └── download_data.py       # downloads the raw dataset with yfinance
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Dataset

- **Source:** Yahoo Finance, downloaded with the `yfinance` library (`scripts/download_data.py`).
- **Ticker:** GOOG (Alphabet Inc.), daily frequency.
- **Period:** 2 January 2019 to 18 September 2026.
- **Size:** 1,939 trading days, 6 columns.
- **Columns:** `Open`, `High`, `Close`, `Low`, `Adj Close`, `Volume`.
- **Download setting:** `auto_adjust=False`, so the raw `Close` and a separate `Adj Close` column are both kept.

Data quality checks showed **no missing values and no duplicate dates**, and all column types were correct (prices are `float64`, volume is `int64`). The raw file is kept untouched in `data/raw_data/`; every later step writes to a different folder.

---

## 5. Phase 1: Exploratory Data Analysis

**Notebook:** `notebooks/1_data_exploration.ipynb`

What was done:

1. Loaded the raw CSV with the date as a datetime index.
2. Inspected structure and statistics (`info()`, `describe()`).
3. Checked for missing values and duplicate dates (none found).
4. Plotted the closing price over time.
5. Plotted the trading volume over time.
6. Plotted the closing price with 50-day and 200-day moving averages (used for visualisation only, not as model features).
7. Saved the cleaned dataset (original 6 columns, sorted by date) to `data/cleaned_data/google_stock_price_cleaned.csv`.

Figures saved to `results/plots/`: `close_price_over_time.png`, `volume_over_time.png`, `moving_averages.png`.

---

## 6. Phase 2: Data Preprocessing

**Notebook:** `notebooks/2_data_preprocessing.ipynb`

### 6.1 Feature selection

The model uses five input features: `Open`, `High`, `Low`, `Close` and `Volume`. `Adj Close` was dropped because it is almost identical to `Close` (it differs only by a small dividend adjustment). The target is the next day's `Close`.

### 6.2 Chronological train / validation / test split

The data is split **by date, with no shuffling**. Shuffling a time series would let the model see the future while training and would make the results look better than they really are.

| Split | Period | Rows | Purpose |
|---|---|---|---|
| Train | 2019-01-02 to 2024-12-31 | 1,510 | The model learns from this data |
| Validation | 2025-01-02 to 2025-12-31 | 250 | Monitors overfitting and decides when to stop training |
| Test | 2026-01-02 to 2026-09-18 | 179 | The most recent, unseen data, used only for the final evaluation |
| **Total** | | **1,939** | |

### 6.3 Scaling

Prices range from about 50 to 400 and volume is in the tens of millions, so the features live on very different scales. A `MinMaxScaler` maps each feature to the range 0 to 1.

The scaler is **fitted on the training data only** and then applied to the validation and test data. Fitting it on the full dataset would leak information from the future into training. The fitted scaler is saved to `saved_models/scaler.pkl` so predictions can be converted back to dollars later.

An important side effect: after scaling, the training data spans exactly 0 to 1, but the validation data reaches about **1.87** and the test data about **2.36**. In other words, the price level in 2025 to 2026 went far above anything seen during training. This distribution shift explains most of the results below.

### 6.4 Sequence creation

An LSTM needs a window of history rather than a single day. For every prediction the model receives the **previous 60 trading days of all 5 features** and predicts the **Close of the following day**. For the validation and test sets, the last 60 rows of the preceding split are prepended as context so that even their first prediction has a full window. This uses only past data, so it does not leak the future.

Final array shapes:

| Split | X (inputs) | y (targets) |
|---|---|---|
| Train | (1450, 60, 5) | (1450,) |
| Validation | (250, 60, 5) | (250,) |
| Test | (179, 60, 5) | (179,) |

The arrays and their target dates are stored as `train.npz`, `val.npz` and `test.npz` in `data/model_ready_data/`.

---

## 7. Phase 3: Model Training and Evaluation

**Notebook:** `notebooks/3_models_training.ipynb`

### 7.1 Model architecture

```
Input (60 timesteps x 5 features)
  -> LSTM(100, return_sequences=True) -> Dropout(0.2)
  -> LSTM(100, return_sequences=True) -> Dropout(0.2)
  -> LSTM(100, return_sequences=True) -> Dropout(0.2)
  -> LSTM(100)                        -> Dropout(0.2)
  -> Dense(1)                         # scaled Close price of the next day
```

- Total parameters: **283,701** (all trainable).
- Dropout of 0.2 randomly switches off 20% of the connections during training to reduce overfitting.

### 7.2 Training setup

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Extra metric | Mean Absolute Error |
| Batch size | 64 |
| Maximum epochs | 200 |
| Early stopping | Monitor `val_loss`, patience 15, restore best weights |
| Random seed | 42 |

### 7.3 Training outcome

- Training stopped automatically at **epoch 37** because the validation loss had not improved for 15 epochs.
- The **best epoch was 22** with a validation loss of **0.0072** (MSE on scaled data). The weights from that epoch were restored and saved to `saved_models/lstm_model.keras`.
- Training loss dropped from 0.0332 in the first epoch to about 0.002 and then stayed flat.

### 7.4 Predictions and metrics

Model outputs are scaled values, so they are converted back to dollars using the saved scaler (the `Close` column is placed inside a dummy 5-column array, inverse-transformed, and extracted again). Three metrics are reported:

- **RMSE:** root mean squared error in USD (penalises large errors more).
- **MAE:** mean absolute error in USD.
- **MAPE:** mean absolute percentage error.

### 7.5 Naive baseline

To check whether the LSTM actually adds value, it is compared against a model-free baseline: **"tomorrow's Close equals today's Close"**. For every window the baseline simply uses the last day's Close as its prediction.

---

## 8. Results

### LSTM model

| Split | RMSE (USD) | MAE (USD) | MAPE |
|---|---|---|---|
| Train | 5.22 | 3.85 | 3.50% |
| Validation | 12.51 | 9.15 | 4.11% |
| Test | 31.68 | 27.25 | 7.78% |

### Naive baseline (previous day's Close)

| Split | RMSE (USD) | MAE (USD) | MAPE |
|---|---|---|---|
| Train | 2.20 | 1.53 | 1.39% |
| Validation | 4.20 | 3.03 | 1.46% |
| Test | 6.88 | 4.87 | 1.43% |

### Comparison (MAE)

| Split | LSTM | Naive baseline |
|---|---|---|
| Train | 3.85 | 1.53 |
| Validation | 9.15 | 3.03 |
| Test | 27.25 | 4.87 |

**The naive baseline beats the LSTM on every split.** On the test set its error is roughly 5.6 times smaller.

---

## 9. Interpreting the Graphs

The figures are stored in `results/plots/`. This section describes what each one shows.

### Closing price over time
GOOG starts at roughly 50 USD in early 2019. There is a sharp but brief drop in early 2020 (the COVID-19 sell-off), followed by a fast recovery and a climb to about 150 USD by the end of 2021. During 2022 the price falls back to below 100 USD. From 2023 the price rises steadily, and after 2025 it accelerates strongly, peaking close to 400 USD in mid-2026 before easing back to the 340 USD area.

### Trading volume over time
Daily volume is noisy and does not show a clear long-term trend. It mostly stays between roughly 10 and 40 million shares, with occasional large spikes that correspond to specific events such as earnings days or big news. There is a visible cluster of high-volume days around March 2020. Volume is on a much more stable scale than price, which is why the features need scaling before training.

### Moving averages
The 50-day average follows the price closely, while the 200-day average is much smoother and reacts slowly. The price falls below both averages during the 2022 decline and moves well above them during the sharp rise after 2025, which confirms the strong momentum in that period. Both lines start later than the price because they need 50 and 200 days of history first.

### Training and validation loss
The training loss falls quickly in the first few epochs and then stays almost flat at a very low value. The validation loss also starts high, falls quickly, but then keeps moving up and down between about 0.007 and 0.024. It reaches its lowest point at epoch 22 (marked on the plot), and never goes lower afterwards, so early stopping ended training at epoch 37. The consistent gap between training and validation loss reflects the change in price level between the two periods.

### Predicted vs. actual price (full timeline)
On the training and validation periods the predicted lines stay very close to the real price. On the test period the predicted line separates from the real one: near the mid-2026 peak the real price is close to 400 USD while the model predicts about 340 USD.

### Predicted vs. actual price (test period, 2026)
The predicted line is smooth and consistently below the actual price. It is roughly 20 USD too low in January, about 55 USD too low around the May peak, and about 30 USD too low in September. It captures the general direction of the moves (down in March to April, up in May, easing afterwards) but misses the size and the day-to-day swings. The model behaves like a slow, damped estimate of the price level rather than a sharp forecast.

---

## 10. Key Findings and Limitations

1. **The LSTM did not beat the naive baseline.** GOOG's price changes by only about 1 to 2% per day, so yesterday's price is already an excellent predictor of today's. Beating it is genuinely hard, and this project measures that honestly.
2. **Distribution shift is the main problem.** The test period contains prices far above anything in the training data (scaled values up to about 2.4 where the scaler saw 0 to 1). The LSTM's internal activations saturate on such inputs, so its predictions are pulled down and flattened.
3. **Predicting the price level is a hard target.** Because the raw price trends upward and its scale changes over time, a model trained on it struggles when the level changes. Predicting returns (daily percentage change) is more stable.
4. **The evaluation is one-step-ahead.** Each prediction uses the previous 60 real days. The project does not forecast several days or months into the future.
5. **Single run, no tuning.** Results come from one training run with a fixed seed and the default architecture. No hyperparameter search was done, so the numbers carry some randomness.
6. **Stock prices depend on many things that are not in the data** (news, earnings, macro events), so the features used here can only ever explain part of the movement.

---

## 11. How to Run

```powershell
# 1. Clone the repository
git clone https://github.com/divyanshu101-ops/basic-stock-price-prediction-model.git
cd basic-stock-price-prediction-model

# 2. Create and activate a virtual environment (Windows PowerShell)
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) register a Jupyter kernel for the notebooks
python -m ipykernel install --user --name stock-lstm --display-name "Python (stock-lstm)"

# 5. Download the raw data (skip if data/raw_data already contains the CSV)
python scripts/download_data.py
```

Then run the notebooks in order, using the `stock-lstm` kernel:

1. `notebooks/1_data_exploration.ipynb`
2. `notebooks/2_data_preprocessing.ipynb`
3. `notebooks/3_models_training.ipynb`

Note: `download_data.py` uses a fixed end date. Change it to download newer data. Because the split dates in Phase 2 are also fixed, update them if you extend the dataset.

---

## 12. Future Work

**Modelling improvements**
- **Predict daily returns instead of prices.** The return range is stable over time, which removes most of the train/test level mismatch and gives the LSTM a fairer task against the baseline.
- **Add technical indicators** (RSI, MACD, Bollinger Bands, moving averages) as extra input features.
- **Tune hyperparameters:** lookback window, number of layers and units, dropout, learning rate and batch size.
- **Try other architectures:** GRU, bidirectional LSTM, CNN-LSTM hybrids, and Transformer-based time-series models.
- **Multi-step forecasting:** predict several days ahead instead of only the next day.
- **Rolling retraining:** periodically retrain on the most recent data so the model adapts to new price levels.

**Better evaluation**
- **Walk-forward validation** instead of a single fixed split.
- **Directional accuracy** (does the model predict up or down correctly?), which is often more meaningful than price error.
- Repeat training with **multiple random seeds** and report mean and standard deviation.
- Compare against additional baselines such as moving-average models, ARIMA or Prophet.

**Data extensions**
- Train on multiple stocks or indices to generalise beyond GOOG.
- Add external signals such as news sentiment, market indices, interest rates or sector data.

**Engineering and deployment**
- Turn the notebooks into a reusable Python package with configuration files.
- Serve the trained model through a **REST API (FastAPI or Flask)** with an endpoint that returns the next-day prediction.
- **Containerise with Docker** and add a scheduled job that downloads fresh data and retrains automatically.
- Track experiments with a tool such as MLflow or Weights & Biases.
- Build a small dashboard (Streamlit) for interactive predictions and plots.

---

## 13. Acknowledgements

- Historical stock data from [Yahoo Finance](https://finance.yahoo.com/) through the `yfinance` library.
- The overall three-phase structure (exploration, preprocessing, model training) and the LSTM architecture were inspired by the open-source project [sinanw/lstm-stock-price-prediction](https://github.com/sinanw/lstm-stock-price-prediction). The dataset, split dates, code and results in this repository are my own.
