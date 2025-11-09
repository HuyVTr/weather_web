# predictor.py
# Trách nhiệm: Tải mô hình đã huấn luyện và cung cấp hàm .predict()
# File này sẽ được controller gọi.

import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/storm_model.pkl')
ml_model = None

def load_model():
    """Tải mô hình vào bộ nhớ."""
    global ml_model
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
            print(f"Tải mô hình ML thành công từ: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Lỗi khi tải mô hình ML: {e}")
            return False
    else:
        print(f"Không tìm thấy file mô hình tại {MODEL_PATH}")
        return False

def predict_storm(input_data):
    """
    Nhận dữ liệu đầu vào (ví dụ: 1 hàng pandas) và trả về dự đoán 24h.
    LƯU Ý: Đây là hàm VÍ DỤ. Dữ liệu đầu vào và đầu ra
    cần khớp với mô hình bạn huấn luyện (ví dụ: LinearRegression)
    """
    if ml_model is None:
        if not load_model():
            return {"error": "Mô hình không khả dụng"}

    try:
        # Extend to predict 24h (simple extrapolation)
        predictions = []
        current_input = input_data.copy()
        for _ in range(24):
            pred = ml_model.predict(current_input)[0]
            predictions.append(pred)
            # Shift input for next hour (simplified)
            current_input.iloc[0, 0] = pred  # Update temp_lag1
            current_input.iloc[0, 1] = current_input.iloc[0, 1] * 0.99  # Pressure decay
            current_input.iloc[0, 2] = current_input.iloc[0, 2] * 1.01  # Wind increase
        
        return {"predicted_temperature": predictions}
    except Exception as e:
        return {"error": f"Lỗi khi dự đoán: {e}"}

# Tải mô hình ngay khi file này được import
load_model()