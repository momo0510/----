# import numpy as np

# # 建立
# np.array([...], dtype=np.float64)
# np.zeros((n, d)); np.ones((n, 1)); np.eye(d)
# np.arange(start, stop, step); np.linspace(a, b, k)

# # 形狀
# A.shape; A.ndim; A.size
# A.reshape(n, d); A.ravel(); A.flatten()

# # 拼接
# np.c_[A, B]                  # 按欄
# np.r_[A, B]                  # 按列
# np.concatenate([A,B], axis=1)

# # 統計
# A.mean(axis=0, keepdims=True)
# A.std(axis=0, keepdims=True)
# A.sum(axis=1)

# # 逐元素/ufunc
# np.exp(A); np.log(A); np.sqrt(A)
# np.maximum(A, 0); A ** 2

# # 線代
# A.T; C = A @ B
# np.linalg.lstsq(X, y, rcond=None)
# np.linalg.pinv(X)
# np.linalg.solve(M, b)
# np.linalg.norm(A)

# # 隨機（可設亂數種子）
# rng = np.random.default_rng(42)
# rng.normal(size=(n,d)); rng.integers(0, 10, size=100)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

start = "2022-01-01"
end = "2023-01-01"

# ------------------------------------------------
# ★ 安全 & 穩定的 Amihud 計算函式（不會報 MultiIndex 錯誤）
# ------------------------------------------------
def compute_amihud(ticker):

    data = yf.download(ticker, start=start, end=end)

    # --- 1. 扁平化欄位（處理 MultiIndex） ---
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]

    # --- 2. 找出唯一的 Close & Volume 欄位 ---
    close_col = [c for c in data.columns if 'Close' in c][0]
    volume_col = [c for c in data.columns if 'Volume' in c][0]

    # --- 3. 建立乾淨的資料框架 ---
    df = pd.DataFrame()
    df['Close'] = data[close_col].astype(float)
    df['Volume'] = data[volume_col].astype(float)

    # --- 4. 計算每日報酬 ---
    df['Return'] = df['Close'].pct_change()

    # --- 5. 計算 Amihud 流動性 ---
    df['Amihud'] = df['Return'].abs() / (df['Volume'] * df['Close'])

    # --- 6. 清理無限值與 NA ---
    df['Amihud'] = df['Amihud'].replace([float('inf'), -float('inf')], None)
    df['Amihud'] = df['Amihud'].fillna(method='bfill')

    return df


# ------------------------------------------------
# ★ 計算 TSLA & QQQ 流動性
# ------------------------------------------------
tsla = compute_amihud("TSLA")
qqq = compute_amihud("QQQ")

# 對齊日期
combined = pd.DataFrame({
    'TSLA': tsla['Amihud'],
    'QQQ': qqq['Amihud']
}).dropna()

# ------------------------------------------------
# ★ Z-score 標準化（核心：比較相對變化）
# ------------------------------------------------
combined['TSLA_Z'] = zscore(combined['TSLA'])
combined['QQQ_Z'] = zscore(combined['QQQ'])

# ------------------------------------------------
# ★ 平滑處理：30天移動平均 (平滑線)
# ------------------------------------------------
combined['TSLA_MA30'] = combined['TSLA_Z'].rolling(30).mean()
combined['QQQ_MA30'] = combined['QQQ_Z'].rolling(30).mean()

# ------------------------------------------------
# ★ 畫圖（平滑線版本）
# ------------------------------------------------
plt.figure(figsize=(16,6))

plt.plot(combined.index, combined['TSLA_MA30'], 
         label="TSLA 30-day MA", color='orange', linewidth=2)

plt.plot(combined.index, combined['QQQ_MA30'], 
         label="QQQ 30-day MA", color='blue', linewidth=2)

# 拆股日期標線
plt.axvline(pd.to_datetime("2022-08-25"), 
            color='red', linestyle='--', linewidth=2, 
            label='Split Date (2022-08-25)')

plt.title("TSLA vs QQQ – Smoothed Amihud Illiquidity (30-day Moving Average, Z-score)")
plt.xlabel("Date")
plt.ylabel("Illiquidity (Z-score, Smoothed)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()





