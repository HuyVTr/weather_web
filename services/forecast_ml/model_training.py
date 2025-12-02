# machine_learning/model_training.py
"""
Training mô hình XGBoost Multi-Output để dự đoán thời tiết
Sử dụng dữ liệu từ database PostgreSQL với schema mới
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*supautils.*")  # chặn supautils luôn
warnings.filterwarnings("ignore")  # chặn tất cả nếu cần

import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.data_forecast.data_storage import connect_to_db

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'weather_xgboost_multi.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data_for_training(conn, province_id=None, limit=200000):
    """
    Load dữ liệu từ database để training – tối ưu tránh timeout.
    """

    # Tắt statement_timeout để tránh query bị kill
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 0;")
    except Exception as e:
        print("⚠️ Không thể set statement_timeout:", e)

    # Query (đã tối ưu: không ORDER BY nếu không cần thiết)
    base_query = """
        SELECT 
            timestamp,
            province_id,
            temperature_2m,
            apparent_temperature,
            relative_humidity_2m,
            precipitation,
            rain,
            showers,
            cloud_cover,
            cloud_cover_low,
            cloud_cover_mid,
            cloud_cover_high,
            weather_code,
            wind_speed_10m,
            wind_direction_10m,
            wind_gusts_10m,
            pressure_msl,
            shortwave_radiation,
            direct_radiation,
            uv_index,
            sunshine_duration
        FROM weather_data
        WHERE temperature_2m IS NOT NULL
        AND pressure_msl IS NOT NULL
    """

    # Điều kiện lọc
    params = []
    if province_id:
        base_query += " AND province_id = %s"
        params.append(province_id)

    # Order theo timestamp DESC để lấy dữ liệu mới nhất
    final_query = base_query + " ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)

    print("⏳ Đang chạy query, vui lòng đợi...")
    df = pd.read_sql(final_query, conn, params=params)

    print(f"✅ Đã tải {len(df)} bản ghi.")

    if len(df) == 0:
        print("⚠️  Không có dữ liệu!")
        return None

    print(f"   • Khoảng thời gian: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   • Số tỉnh: {df['province_id'].nunique()}")
    return df


def feature_engineering(df):
    """
    Tạo features cho model
    Bao gồm: lag features, rolling features, time features
    
    Args:
        df: DataFrame từ database
    
    Returns:
        X: Features
        y: Targets
        feature_cols: Danh sách tên các features
    """
    print("\n🔧 Đang tạo features...")
    
    df = df.sort_values(['province_id', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Fill missing values với giá trị hợp lý
    fill_values = {
        'apparent_temperature': df['temperature_2m'],  # Nếu thiếu thì dùng temp thường
        'precipitation': 0,
        'rain': 0,
        'showers': 0,
        'cloud_cover': 50,
        'cloud_cover_low': 0,
        'cloud_cover_mid': 0,
        'cloud_cover_high': 0,
        'wind_gusts_10m': df['wind_speed_10m'],  # Nếu thiếu gust thì dùng wind speed
        'shortwave_radiation': 0,
        'direct_radiation': 0,
        'uv_index': 0,
        'sunshine_duration': 0,
        'weather_code': 1
    }
    
    for col, default_val in fill_values.items():
        if col in df.columns:
            df[col].fillna(default_val, inplace=True)
    
    # =========================================================================
    # LAG FEATURES - Dữ liệu từ các giờ trước
    # =========================================================================
    print("   📌 Tạo lag features...")
    lags = [1, 2, 3, 6, 12, 24]  # 1h, 2h, 3h, 6h, 12h, 24h trước
    
    lag_cols = [
        'temperature_2m',
        'relative_humidity_2m', 
        'wind_speed_10m',
        'pressure_msl',
        'precipitation',
        'cloud_cover'
    ]
    
    for lag in lags:
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby('province_id')[col].shift(lag)
    
    # =========================================================================
    # ROLLING FEATURES - Trung bình trượt
    # =========================================================================
    print("   📌 Tạo rolling features...")
    rolls = [3, 6, 24]  # 3h, 6h, 24h
    
    for w in rolls:
        # Temperature
        df[f'temp_roll_mean_{w}'] = df.groupby('province_id')['temperature_2m'].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f'temp_roll_std_{w}'] = df.groupby('province_id')['temperature_2m'].transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )
        df[f'temp_roll_min_{w}'] = df.groupby('province_id')['temperature_2m'].transform(
            lambda x: x.rolling(w, min_periods=1).min()
        )
        df[f'temp_roll_max_{w}'] = df.groupby('province_id')['temperature_2m'].transform(
            lambda x: x.rolling(w, min_periods=1).max()
        )
        
        # Precipitation - Tổng lượng mưa
        df[f'precip_roll_sum_{w}'] = df.groupby('province_id')['precipitation'].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )
        
        # Humidity
        df[f'humidity_roll_mean_{w}'] = df.groupby('province_id')['relative_humidity_2m'].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        
        # Pressure
        df[f'pressure_roll_mean_{w}'] = df.groupby('province_id')['pressure_msl'].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        
        # Wind
        df[f'wind_roll_mean_{w}'] = df.groupby('province_id')['wind_speed_10m'].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f'wind_roll_max_{w}'] = df.groupby('province_id')['wind_speed_10m'].transform(
            lambda x: x.rolling(w, min_periods=1).max()
        )
    
    # =========================================================================
    # TIME FEATURES - Đặc trưng thời gian
    # =========================================================================
    print("   📌 Tạo time features...")
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Season (mùa)
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,  # Đông
        3: 1, 4: 1, 5: 1,   # Xuân
        6: 2, 7: 2, 8: 2,   # Hè
        9: 3, 10: 3, 11: 3  # Thu
    })
    
    # Time of day (buổi trong ngày)
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)   

    
    # =========================================================================
    # INTERACTION FEATURES - Tương tác giữa các biến
    # =========================================================================
    print("   📌 Tạo interaction features...")
    
    # Nhiệt độ x Độ ẩm (cảm giác oi bức)
    df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m'] / 100
    
    # Nhiệt độ x Gió (cảm giác lạnh do gió)
    df['temp_wind_interaction'] = df['temperature_2m'] * df['wind_speed_10m']
    
    # Áp suất x Độ ẩm (khả năng mưa)
    df['pressure_humidity_interaction'] = df['pressure_msl'] * df['relative_humidity_2m'] / 100
    
    # Cloud cover total (tổng độ che phủ mây)
    if all(col in df.columns for col in ['cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']):
        df['cloud_cover_total'] = df['cloud_cover_low'] + df['cloud_cover_mid'] + df['cloud_cover_high']
    
    # =========================================================================
    # TARGET VARIABLES - Dự đoán 1 giờ tới
    # =========================================================================
    print("   📌 Tạo target variables...")
    
    target_cols = [
        'temperature_2m',           # Nhiệt độ
        'relative_humidity_2m',     # Độ ẩm
        'precipitation',            # Lượng mưa
        'wind_speed_10m',           # Tốc độ gió
        'pressure_msl',             # Áp suất
        'cloud_cover'               # Độ phủ mây (thay vì visibility)
    ]
    
    # Tạo target cho 1 giờ tới
    for col in target_cols:
        if col in df.columns:
            df[f'{col}_next'] = df.groupby('province_id')[col].shift(-1)
    
    # =========================================================================
    # DROP ROWS WITH MISSING VALUES
    # =========================================================================
    print("   📌 Loại bỏ dữ liệu thiếu...")
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    print(f"   ✅ Đã loại bỏ {initial_rows - final_rows} dòng có giá trị thiếu")
    
    # =========================================================================
    # SELECT FEATURES AND TARGETS
    # =========================================================================
    # Các cột không dùng làm feature
    exclude_cols = [
        'timestamp', 'province_id', 'weather_code', 
        'apparent_temperature',  # Đã có temp_humidity_interaction
        'rain', 'showers',  # Đã có precipitation
        'wind_direction_10m',  # Direction không quan trọng bằng speed
        'wind_gusts_10m',  # Đã có wind_roll_max
        'shortwave_radiation', 'direct_radiation',  # Tương quan cao với sunshine_duration
        'sunshine_duration',  # Có thể bỏ nếu không cần thiết
        'uv_index',  # UV có thể tính từ hour và month
    ] + target_cols + [f'{c}_next' for c in target_cols]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target_cols_next = [f'{c}_next' for c in target_cols]
    
    X = df[feature_cols]
    y = df[target_cols_next]
    
    print(f"\n✅ Feature engineering hoàn tất!")
    print(f"   • Số features: {len(feature_cols)}")
    print(f"   • Số samples: {len(X)}")
    print(f"   • Target variables: {', '.join(target_cols)}")
    
    return X, y, feature_cols

def train(province_id=None, save_path=MODEL_PATH):
    """
    Huấn luyện mô hình XGBoost Multi-Output
    
    Args:
        province_id: Nếu None thì train cho tất cả tỉnh, nếu có giá trị thì chỉ train cho tỉnh đó
        save_path: Đường dẫn lưu model
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH XGBOOST MULTI-OUTPUT")
    print("="*80)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n[BƯỚC 1/5] 📥 Đang tải dữ liệu từ database...")
    conn = connect_to_db() 
    df = load_data_for_training(conn, province_id=province_id, limit=500000)
    conn.close()
    if df is None or len(df) < 1000:
        print("\n❌ THẤT BẠI: Không đủ dữ liệu để huấn luyện (cần ít nhất 1000 bản ghi)")
        print("💡 Vui lòng chạy data collection để thu thập dữ liệu:")
        print("   python data_pipeline/data_collection.py")
        return False
    
    # =========================================================================
    # 2. FEATURE ENGINEERING
    # =========================================================================
    print("\n[BƯỚC 2/5] 🔧 Đang xử lý và tạo features...")
    try:
        X, y, feature_cols = feature_engineering(df)
    except Exception as e:
        print(f"\n❌ THẤT BẠI: Lỗi khi tạo features: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if len(X) < 100:
        print("\n❌ THẤT BẠI: Không đủ dữ liệu sau khi xử lý")
        return False
    
    # =========================================================================
    # 3. SPLIT DATA
    # =========================================================================
    print("\n[BƯỚC 3/5] ✂️  Đang chia dữ liệu train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        shuffle=False  # Giữ thứ tự thời gian
    )
    
    print(f"   ✅ Train size: {len(X_train):,} samples")
    print(f"   ✅ Test size:  {len(X_test):,} samples")
    print(f"   ✅ Features:   {len(feature_cols)}")
    
    # =========================================================================
    # 4. TRAIN MODEL
    # =========================================================================
    print("\n[BƯỚC 4/5] 🤖 Đang huấn luyện mô hình XGBoost...")
    print("   ⏳ Quá trình này có thể mất vài phút...")
    
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=1000,          # Số cây quyết định
            learning_rate=0.05,         # Tốc độ học
            max_depth=10,               # Độ sâu tối đa của cây
            subsample=0.8,              # Tỷ lệ mẫu con
            colsample_bytree=0.8,       # Tỷ lệ feature cho mỗi cây
            random_state=42,
            n_jobs=-1,                  # Sử dụng tất cả CPU
            tree_method='hist',         # Faster training
            min_child_weight=3,         # Regularization
            gamma=0.1,                  # Regularization
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            verbosity=0,                 # Tắt log của XGBoost
            enable_categorical=True
        )
    )
    
    try:
        model.fit(X_train, y_train)
        print("   ✅ Hoàn thành huấn luyện!")
    except Exception as e:
        print(f"\n❌ THẤT BẠI: Lỗi khi train model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # 5. EVALUATE MODEL
    # =========================================================================
    print("\n[BƯỚC 5/5] 📊 Đang đánh giá mô hình trên tập test...")
    
    try:
        pred = model.predict(X_test)
    except Exception as e:
        print(f"\n❌ THẤT BẠI: Lỗi khi predict: {e}")
        return False
    
    target_names = [
        '🌡️  Nhiệt độ (°C)', 
        '💧 Độ ẩm (%)', 
        '🌧️  Lượng mưa (mm)', 
        '🌬️  Tốc độ gió (km/h)', 
        '🔵 Áp suất (hPa)', 
        '☁️  Độ phủ mây (%)'
    ]
    
    print("\n" + "="*80)
    print("📈 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Dự đoán 1 giờ tới)")
    print("="*80)
    
    overall_score = 0
    scores = []
    
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(y_test.iloc[:, i], pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], pred[:, i])
        
        print(f"\n{name}:")
        print(f"   • MAE:  {mae:.3f}")
        print(f"   • RMSE: {rmse:.3f}")
        print(f"   • R²:   {r2:.3f}", end="")
        
        # Đánh giá chất lượng
        if r2 > 0.90:
            print(" ✅ XUẤT SẮC")
            scores.append(3)
        elif r2 > 0.80:
            print(" ✅ TỐT")
            scores.append(2)
        elif r2 > 0.70:
            print(" ⚠️  CHẤP NHẬN ĐƯỢC")
            scores.append(1)
        else:
            print(" ❌ CẦN CẢI THIỆN")
            scores.append(0)
    
    overall_score = np.mean(scores)
    
    # =========================================================================
    # 6. SAVE MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("💾 ĐANG LƯU MÔ HÌNH...")
    
    try:
        joblib.dump(model, save_path)
        joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_cols.pkl'))
        print(f"✅ Đã lưu mô hình tại: {save_path}")
        print(f"✅ Đã lưu feature columns tại: {os.path.join(MODEL_DIR, 'feature_cols.pkl')}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu model: {e}")
        return False
    
    # =========================================================================
    # 7. FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("🎯 TỔNG KẾT")
    print("="*80)
    print(f"Điểm trung bình: {overall_score:.2f}/3.0")
    
    if overall_score >= 2.5:
        print("\n🌟🌟🌟 MÔ HÌNH XUẤT SẮC - ĐỘ TIN CẬY RẤT CAO!")
        print("✅ Có thể sử dụng cho dự đoán thời tiết thực tế")
    elif overall_score >= 1.5:
        print("\n⭐⭐ MÔ HÌNH TỐT - ĐỘ TIN CẬY CHẤP NHẬN ĐƯỢC")
        print("✅ Phù hợp cho ứng dụng thực tế")
    elif overall_score >= 1.0:
        print("\n⚠️  MÔ HÌNH TRUNG BÌNH - NÊN CẢI THIỆN")
        print("💡 Thu thập thêm dữ liệu và điều chỉnh hyperparameters")
    else:
        print("\n❌ MÔ HÌNH YẾU - CẦN CẢI THIỆN")
        print("💡 Cần thu thập nhiều dữ liệu hơn (ít nhất 10,000 samples)")
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT HUẤN LUYỆN!")
    print("="*80)
    
    print("\n📝 Bạn có thể sử dụng mô hình bằng cách:")
    print("   from machine_learning.predictor import predict_storm")
    print("   result = predict_storm(province_id=1)")
    print("\n💡 Để test độ chính xác:")
    print("   python test_ml_accuracy.py")
    print("   python evaluate_dashboard.py")
    print()
    
    return True

def retrain_for_province(province_id):
    """
    Huấn luyện lại mô hình cho một tỉnh cụ thể
    
    Args:
        province_id: ID của tỉnh cần train
    
    Returns:
        bool: True nếu thành công
    """
    model_path = os.path.join(MODEL_DIR, f'weather_xgboost_province_{province_id}.pkl')
    return train(province_id=province_id, save_path=model_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train mô hình dự đoán thời tiết')
    parser.add_argument('--province_id', type=int, help='ID tỉnh cần train (bỏ qua để train tất cả)')
    
    args = parser.parse_args()
    
    # Huấn luyện mô hình
    success = train(province_id=args.province_id)
    
    if not success:
        print("\n❌ Training thất bại!")
        sys.exit(1)
    
    sys.exit(0)