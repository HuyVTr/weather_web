# model_training.py
# Trách nhiệm: Tải dữ liệu, huấn luyện, và lưu model.
import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime  # Thêm import

# Thêm thư mục gốc vào path để import data_storage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.data_storage import connect_to_db

# Đường dẫn lưu mô hình
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'storm_model.pkl')

def load_data_for_training(province_id=None):
    """Tải dữ liệu mẫu từ DB để huấn luyện."""
    print("Đang tải dữ liệu huấn luyện từ DB...")
    conn = connect_to_db()
    if not conn:
        return pd.DataFrame()
    try:
        # Ví dụ: Lấy dữ liệu của tỉnh cụ thể hoặc tất cả
        query = """
            SELECT "timestamp", temperature_2m, pressure_msl, wind_speed_10m
            FROM weather_data
            WHERE province_id = COALESCE(:province_id, province_id) AND pressure_msl IS NOT NULL
            ORDER BY "timestamp" DESC
            LIMIT 50000
        """
        df = pd.read_sql(query, conn, params={'province_id': province_id})
        conn.close()
        print(f"Tải thành công {len(df)} dòng dữ liệu mẫu.")

        # Nếu <24h, lấy thêm dữ liệu ngày trước
        if len(df) < 24:
            conn = connect_to_db()  # Mở kết nối mới
            if conn:
                start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                extended_query = """
                    SELECT "timestamp", temperature_2m, pressure_msl, wind_speed_10m
                    FROM weather_data
                    WHERE province_id = :province_id AND timestamp >= :start_date AND pressure_msl IS NOT NULL
                    ORDER BY "timestamp" DESC
                    LIMIT 50000
                """
                extended_df = pd.read_sql(extended_query, conn, params={'province_id': province_id, 'start_date': start_date})
                df = pd.concat([df, extended_df]).drop_duplicates().sort_values('timestamp', ascending=False)
                conn.close()
                print(f"Đã mở rộng dữ liệu với {len(extended_df)} dòng từ ngày trước.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def feature_engineering(df):
    """Tạo features và targets."""
    print("Đang xử lý feature engineering...")
    df['temp_lag1'] = df['temperature_2m'].shift(1)
    df['pressure_lag1'] = df['pressure_msl'].shift(1)
    df['wind_lag1'] = df['wind_speed_10m'].shift(1)
    df['temp_target'] = df['temperature_2m'].shift(-1)
    df.dropna(inplace=True)
    
    features = ['temp_lag1', 'pressure_lag1', 'wind_lag1']
    targets = ['temp_target']
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    return df[features], df[targets]

def train_model(X, y):
    """Huấn luyện mô hình."""
    print("Đang huấn luyện mô hình (Linear Regression ví dụ)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Trả về model và test data để đánh giá
    return model, X_test, y_test

def save_model(model):
    """Lưu mô hình vào file."""
    print(f"Đang lưu mô hình vào: {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)  # Tạo thư mục 'models'
    joblib.dump(model, MODEL_PATH)
    print("Lưu mô hình thành công!")

if __name__ == "__main__":
    df = load_data_for_training()
    if not df.empty:
        X, y = feature_engineering(df)
        if not X.empty:
            model, X_test, y_test = train_model(X, y)
            save_model(model)
            print("\n--- QUY TRÌNH HUẤN LUYỆN HOÀN TẤT ---")