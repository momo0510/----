import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import os

class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

class LinearRegressionSGD(LinearRegressionBase):
    def fit(self, X, y, learning_rate=0.01, T=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0
        
        # T 次更新
        for t in range(T):
            # 隨機選點
            idx = np.random.randint(0, n_samples)
            xi = X[idx]
            yi = y[idx]

            # 值和誤差 
            y_pred = np.dot(xi, self.weights) + self.intercept
            error = y_pred - yi

            # 更新參數
            self.weights -= learning_rate * (error * xi)
            self.intercept -= learning_rate * error

            if (t + 1) % 200 == 0:
                logger.info(f"Iteration {t+1}/{T} completed.")

    def predict(self, X):
        return X @ self.weights + self.intercept

def compute_mse(prediction, ground_truth):
    return np.mean((prediction - ground_truth) ** 2)

def main():
    current_dir = os.path.dirname(__file__)
    
    try:
        x_df = pd.read_csv(os.path.join(current_dir, 'Averaged homework scores.csv'))
        y_df = pd.read_csv(os.path.join(current_dir, 'Final exam scores.csv'))
        
        X_all = x_df.values.reshape(-1, 1) 
        y_all = y_df.values.reshape(-1)
    except Exception as e:
        logger.error(f"讀取失敗: {e}")
        return

    # 分割原始資料
    train_x_orig, test_x_orig = X_all[:400], X_all[400:]
    train_y, test_y = y_all[:400], y_all[400:]

    # 資料縮放：全部除以 100
    train_x = train_x_orig / 100.0
    test_x = test_x_orig / 100.0

    model = LinearRegressionSGD()
    model.fit(train_x, train_y, learning_rate=0.01, T=1000)

    y_test_pred = model.predict(test_x)
    test_mse = compute_mse(y_test_pred, test_y)
    
    logger.info(f"Final Weights: {model.weights}")
    logger.info(f"Final Intercept: {model.intercept:.4f}")
    logger.info(f"Testing MSE: {test_mse:.4f}")

    # 繪圖 
    plt.figure(figsize=(10, 6))
    # 畫點：使用原始 X 座標 (0~100)
    plt.scatter(test_x_orig, test_y, color='black', alpha=0.6, marker='*', s=80, linewidths=1.5, label='Testing dataset')
    
    line_x_orig = np.linspace(test_x_orig.min(), test_x_orig.max(), 100).reshape(-1, 1)
    line_y = model.predict(line_x_orig / 100.0)
    
    plt.plot(line_x_orig, line_y, color='red', linewidth=2, label='Linear regression result')
    
    plt.xlabel('Averaged Homework Scores')
    plt.ylabel('Final Exam Scores')
    plt.title(f'Part 1: SGD Linear Regression (Testing MSE: {test_mse:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()