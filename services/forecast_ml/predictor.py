# machine_learning/predictor.py
"""
Dự đoán thời tiết sử dụng mô hình XGBoost đã train
Tương thích với database schema mới
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.data_storage import connect_to_db

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/weather_xgboost_multi.pkl')
FEATURE_COLS_PATH = os.path.join(os.path.dirname(__file__), 'models/feature_cols.pkl')

model = None
feature_cols = None

def load_model():
    """Load mô hình ML đã được train"""
    global model, feature_cols
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_COLS_PATH):
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        print("✅ ĐÃ TẢI MÔ HÌNH XGBOOST THÀNH CÔNG")
        return True
    print("⚠️  KHÔNG TÌM THẤY MÔ HÌNH")
    print("💡 Vui lòng chạy: python machine_learning/model_training.py")
    return False

def load_historical_data(province_id, hours=168):
    """
    Lấy dữ liệu lịch sử từ database (theo schema mới)
    
    Args:
        province_id: ID của tỉnh
        hours: Số giờ lịch sử cần lấy (mặc định 168 = 7 ngày)
    
    Returns:
        DataFrame với dữ liệu lịch sử
    """
    conn = connect_to_db()
    query = """
        SELECT 
            timestamp,
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
        WHERE province_id = %s 
        ORDER BY timestamp DESC 
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(province_id, hours))
    conn.close()
    
    # Sort theo thứ tự thời gian tăng dần
    df = df.sort_values('timestamp')
    
    # Fill missing values
    fill_values = {
        'apparent_temperature': df['temperature_2m'],
        'precipitation': 0,
        'rain': 0,
        'showers': 0,
        'cloud_cover': 50,
        'cloud_cover_low': 0,
        'cloud_cover_mid': 0,
        'cloud_cover_high': 0,
        'wind_gusts_10m': df['wind_speed_10m'],
        'shortwave_radiation': 0,
        'direct_radiation': 0,
        'uv_index': 0,
        'sunshine_duration': 0,
        'weather_code': 1
    }
    
    for col, default_val in fill_values.items():
        if col in df.columns:
            df[col].fillna(default_val, inplace=True)
    
    return df

def create_features(df, target_time):
    """
    Tạo features từ dữ liệu lịch sử cho một thời điểm dự đoán
    
    Args:
        df: DataFrame chứa dữ liệu lịch sử
        target_time: Thời điểm cần dự đoán
    
    Returns:
        DataFrame với 1 dòng chứa features
    """
    if len(df) < 24:
        return None
    
    # Lấy dòng cuối cùng làm base
    row = df.tail(1).copy()
    
    # =========================================================================
    # LAG FEATURES
    # =========================================================================
    lags = [1, 2, 3, 6, 12, 24]
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
            if len(df) >= lag and col in df.columns:
                row[f'{col}_lag{lag}'] = df[col].iloc[-lag]
            else:
                row[f'{col}_lag{lag}'] = row[col].iloc[0]
    
    # =========================================================================
    # ROLLING FEATURES
    # =========================================================================
    rolls = [3, 6, 24]
    
    for w in rolls:
        window = min(w, len(df))
        
        # Temperature
        row[f'temp_roll_mean_{w}'] = df['temperature_2m'].tail(window).mean()
        row[f'temp_roll_std_{w}'] = df['temperature_2m'].tail(window).std() if window > 1 else 0
        row[f'temp_roll_min_{w}'] = df['temperature_2m'].tail(window).min()
        row[f'temp_roll_max_{w}'] = df['temperature_2m'].tail(window).max()
        
        # Precipitation
        row[f'precip_roll_sum_{w}'] = df['precipitation'].tail(window).sum()
        
        # Humidity
        row[f'humidity_roll_mean_{w}'] = df['relative_humidity_2m'].tail(window).mean()
        
        # Pressure
        row[f'pressure_roll_mean_{w}'] = df['pressure_msl'].tail(window).mean()
        
        # Wind
        row[f'wind_roll_mean_{w}'] = df['wind_speed_10m'].tail(window).mean()
        row[f'wind_roll_max_{w}'] = df['wind_speed_10m'].tail(window).max()
    
    # =========================================================================
    # TIME FEATURES
    # =========================================================================
    row['hour'] = target_time.hour
    row['dayofweek'] = target_time.weekday()
    row['month'] = target_time.month
    row['day'] = target_time.day
    row['is_weekend'] = 1 if target_time.weekday() >= 5 else 0
    
    # Season
    season_map = {
        12: 0, 1: 0, 2: 0,  # Đông
        3: 1, 4: 1, 5: 1,   # Xuân
        6: 2, 7: 2, 8: 2,   # Hè
        9: 3, 10: 3, 11: 3  # Thu
    }
    row['season'] = season_map.get(target_time.month, 0)
    
    # Time of day
    if 0 <= target_time.hour < 6:
        row['time_of_day'] = 0  # Đêm
    elif 6 <= target_time.hour < 12:
        row['time_of_day'] = 1  # Sáng
    elif 12 <= target_time.hour < 18:
        row['time_of_day'] = 2  # Chiều
    else:
        row['time_of_day'] = 3  # Tối
    
    # =========================================================================
    # INTERACTION FEATURES
    # =========================================================================
    row['temp_humidity_interaction'] = row['temperature_2m'].iloc[0] * row['relative_humidity_2m'].iloc[0] / 100
    row['temp_wind_interaction'] = row['temperature_2m'].iloc[0] * row['wind_speed_10m'].iloc[0]
    row['pressure_humidity_interaction'] = row['pressure_msl'].iloc[0] * row['relative_humidity_2m'].iloc[0] / 100
    
    # Cloud cover total
    if all(col in row.columns for col in ['cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']):
        row['cloud_cover_total'] = (row['cloud_cover_low'].iloc[0] + 
                                    row['cloud_cover_mid'].iloc[0] + 
                                    row['cloud_cover_high'].iloc[0])
    
    return row

def predict_weather_code(temp, precip, humidity, wind, cloud_cover):
    """
    Dự đoán weather code dựa trên các yếu tố thời tiết
    
    Returns:
        int: Weather code
    """
    # Giông bão
    if wind > 20 and precip > 5:
        return 95  # Thunderstorm
    # Mưa to
    elif precip > 10:
        return 65  # Heavy rain
    elif precip > 5:
        return 63  # Moderate rain
    # Mưa nhẹ/phùn
    elif precip > 2:
        return 61  # Light rain
    elif precip > 0.5:
        return 51  # Light drizzle
    # Sương mù
    elif humidity > 90 and temp < 20:
        return 45  # Fog
    # Nhiều mây
    elif cloud_cover > 80:
        return 3  # Overcast
    # Mây rải rác
    elif cloud_cover > 50:
        return 2  # Partly cloudy
    # Ít mây
    elif temp > 28 or cloud_cover > 20:
        return 1  # Mainly clear
    # Quang đãng
    else:
        return 0  # Clear sky

def predict_uv_index(hour, month, weather_code):
    """
    Dự đoán chỉ số UV dựa trên giờ, tháng và thời tiết
    
    Returns:
        float: UV index
    """
    # UV cao nhất vào giữa trưa
    if 11 <= hour <= 14:
        base_uv = 8
    elif 9 <= hour <= 16:
        base_uv = 6
    elif 7 <= hour <= 17:
        base_uv = 3
    else:
        return 0
    
    # Điều chỉnh theo tháng (mùa hè cao hơn)
    if month in [5, 6, 7, 8]:
        base_uv += 2
    elif month in [3, 4, 9, 10]:
        base_uv += 1
    
    # Điều chỉnh theo thời tiết
    if weather_code in [61, 63, 65, 95]:  # Mưa/giông
        base_uv *= 0.3
    elif weather_code in [3, 45]:  # Nhiều mây/sương mù
        base_uv *= 0.5
    elif weather_code in [2]:  # Mây rải rác
        base_uv *= 0.7
    
    return min(max(round(base_uv, 1), 0), 11)

def calculate_visibility(humidity, precipitation, cloud_cover):
    """
    Ước tính tầm nhìn dựa trên độ ẩm, mưa và mây
    
    Returns:
        int: Visibility in meters
    """
    base_visibility = 50000  # 50km
    
    # Giảm tầm nhìn khi mưa
    if precipitation > 10:
        base_visibility = 2000
    elif precipitation > 5:
        base_visibility = 5000
    elif precipitation > 1:
        base_visibility = 10000
    
    # Giảm tầm nhìn khi độ ẩm cao
    if humidity > 95:
        base_visibility = min(base_visibility, 5000)
    elif humidity > 90:
        base_visibility = min(base_visibility, 10000)
    
    # Giảm tầm nhìn khi nhiều mây
    if cloud_cover > 90:
        base_visibility = min(base_visibility, 15000)
    
    return int(base_visibility)

def predict_storm(province_id, current_weather_data=None):
    """
    Dự đoán thời tiết cho 24 giờ tới (hourly) và 7 ngày tới (daily)
    
    Args:
        province_id: ID của tỉnh cần dự đoán
        current_weather_data: Dict với dữ liệu hiện tại (fallback nếu DB thiếu)
    
    Returns:
        dict với các key:
        - predicted_temperature: list 24 giá trị nhiệt độ
        - predicted_humidity: list 24 giá trị độ ẩm
        - predicted_precipitation: list 24 giá trị lượng mưa
        - predicted_wind_speed: list 24 giá trị tốc độ gió
        - predicted_pressure: list 24 giá trị áp suất
        - predicted_cloud_cover: list 24 giá trị độ phủ mây
        - predicted_weather_code: list 24 mã thời tiết
        - predicted_uv_index: list 24 chỉ số UV
        - predicted_visibility: list 24 tầm nhìn
        - daily_forecast: list 7 dict với thông tin mỗi ngày
        - prediction_hours: 24
    """
    if model is None and not load_model():
        return {"error": "Không thể load mô hình"}
    
    try:
        # Load dữ liệu lịch sử
        df = load_historical_data(province_id, hours=168)
        
        if len(df) < 24:
            # Fallback: dùng current weather data
            if current_weather_data:
                df = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'temperature_2m': current_weather_data.get('temperature_2m', 25),
                    'apparent_temperature': current_weather_data.get('temperature_2m', 25),
                    'relative_humidity_2m': current_weather_data.get('relative_humidity_2m', 70),
                    'precipitation': 0,
                    'rain': 0,
                    'showers': 0,
                    'cloud_cover': 50,
                    'cloud_cover_low': 20,
                    'cloud_cover_mid': 20,
                    'cloud_cover_high': 10,
                    'weather_code': 1,
                    'wind_speed_10m': current_weather_data.get('wind_speed_10m', 5),
                    'wind_direction_10m': 0,
                    'wind_gusts_10m': current_weather_data.get('wind_speed_10m', 5),
                    'pressure_msl': current_weather_data.get('pressure_msl', 1013),
                    'shortwave_radiation': 0,
                    'direct_radiation': 0,
                    'uv_index': 0,
                    'sunshine_duration': 0
                }])
            else:
                return {"error": "Không đủ dữ liệu lịch sử"}
        
        # Dự đoán 24 giờ tới (hourly)
        hourly_predictions = []
        current_df = df.copy()
        now = datetime.now()
        
        for hour in range(24):
            target_time = now + timedelta(hours=hour + 1)
            
            # Tạo features
            feature_row = create_features(current_df, target_time)
            if feature_row is None:
                break
            
            # Predict
            X = feature_row[feature_cols]
            pred = model.predict(X)[0]
            
            # Lấy kết quả (6 targets: temp, humidity, precip, wind, pressure, cloud_cover)
            temp = float(pred[0])
            humidity = int(np.clip(pred[1], 0, 100))
            precip = max(float(pred[2]), 0)
            wind = max(float(pred[3]), 0)
            pressure = float(pred[4])
            cloud_cover = float(np.clip(pred[5], 0, 100))
            
            # Dự đoán các giá trị khác
            weather_code = predict_weather_code(temp, precip, humidity, wind, cloud_cover)
            uv_index = predict_uv_index(target_time.hour, target_time.month, weather_code)
            visibility = calculate_visibility(humidity, precip, cloud_cover)
            
            hourly_predictions.append({
                'temperature_2m': round(temp, 1),
                'relative_humidity_2m': humidity,
                'precipitation': round(precip, 2),
                'wind_speed_10m': round(wind, 1),
                'pressure_msl': round(pressure, 1),
                'cloud_cover': round(cloud_cover, 1),
                'weather_code': weather_code,
                'uv_index': uv_index,
                'visibility': visibility,
                'time': target_time.isoformat()
            })
            
            # Cập nhật df cho prediction tiếp theo
            new_row = pd.DataFrame([{
                'timestamp': target_time,
                'temperature_2m': temp,
                'apparent_temperature': temp,
                'relative_humidity_2m': humidity,
                'precipitation': precip,
                'rain': precip,
                'showers': 0,
                'cloud_cover': cloud_cover,
                'cloud_cover_low': cloud_cover / 3,
                'cloud_cover_mid': cloud_cover / 3,
                'cloud_cover_high': cloud_cover / 3,
                'weather_code': weather_code,
                'wind_speed_10m': wind,
                'wind_direction_10m': 0,
                'wind_gusts_10m': wind * 1.2,
                'pressure_msl': pressure,
                'shortwave_radiation': 0,
                'direct_radiation': 0,
                'uv_index': uv_index,
                'sunshine_duration': 0
            }])
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # Tổng hợp daily forecast (7 ngày)
        daily_forecast = []
        for day in range(7):
            start_idx = day * 24
            end_idx = start_idx + 24
            
            # Predict thêm nếu chưa đủ
            while len(hourly_predictions) < end_idx:
                target_time = now + timedelta(hours=len(hourly_predictions) + 1)
                feature_row = create_features(current_df, target_time)
                if feature_row is None:
                    break
                
                X = feature_row[feature_cols]
                pred = model.predict(X)[0]
                
                temp = float(pred[0])
                humidity = int(np.clip(pred[1], 0, 100))
                precip = max(float(pred[2]), 0)
                wind = max(float(pred[3]), 0)
                pressure = float(pred[4])
                cloud_cover = float(np.clip(pred[5], 0, 100))
                weather_code = predict_weather_code(temp, precip, humidity, wind, cloud_cover)
                uv_index = predict_uv_index(target_time.hour, target_time.month, weather_code)
                visibility = calculate_visibility(humidity, precip, cloud_cover)
                
                hourly_predictions.append({
                    'temperature_2m': round(temp, 1),
                    'relative_humidity_2m': humidity,
                    'precipitation': round(precip, 2),
                    'wind_speed_10m': round(wind, 1),
                    'pressure_msl': round(pressure, 1),
                    'cloud_cover': round(cloud_cover, 1),
                    'weather_code': weather_code,
                    'uv_index': uv_index,
                    'visibility': visibility,
                    'time': target_time.isoformat()
                })
                
                new_row = pd.DataFrame([{
                    'timestamp': target_time,
                    'temperature_2m': temp,
                    'apparent_temperature': temp,
                    'relative_humidity_2m': humidity,
                    'precipitation': precip,
                    'rain': precip,
                    'showers': 0,
                    'cloud_cover': cloud_cover,
                    'cloud_cover_low': cloud_cover / 3,
                    'cloud_cover_mid': cloud_cover / 3,
                    'cloud_cover_high': cloud_cover / 3,
                    'weather_code': weather_code,
                    'wind_speed_10m': wind,
                    'wind_direction_10m': 0,
                    'wind_gusts_10m': wind * 1.2,
                    'pressure_msl': pressure,
                    'shortwave_radiation': 0,
                    'direct_radiation': 0,
                    'uv_index': uv_index,
                    'sunshine_duration': 0
                }])
                current_df = pd.concat([current_df, new_row], ignore_index=True)
            
            day_data = hourly_predictions[start_idx:end_idx]
            if not day_data:
                continue
            
            temps = [h['temperature_2m'] for h in day_data]
            precips = [h['precipitation'] for h in day_data]
            winds = [h['wind_speed_10m'] for h in day_data]
            weather_codes = [h['weather_code'] for h in day_data]
            
            # Tính toán sunrise/sunset
            day_date = (now + timedelta(days=day)).date()
            sunrise = datetime.combine(day_date, datetime.min.time().replace(hour=6, minute=0))
            sunset = datetime.combine(day_date, datetime.min.time().replace(hour=18, minute=0))
            
            daily_forecast.append({
                'time': day_date.isoformat(),
                'temperature_2m_max': round(max(temps), 1),
                'temperature_2m_min': round(min(temps), 1),
                'precipitation_sum': round(sum(precips), 2),
                'wind_speed_10m_max': round(max(winds), 1),
                'weather_code': max(set(weather_codes), key=weather_codes.count),
                'sunrise': sunrise.isoformat(),
                'sunset': sunset.isoformat()
            })
        
        return {
            'predicted_temperature': [h['temperature_2m'] for h in hourly_predictions[:24]],
            'predicted_humidity': [h['relative_humidity_2m'] for h in hourly_predictions[:24]],
            'predicted_precipitation': [h['precipitation'] for h in hourly_predictions[:24]],
            'predicted_wind_speed': [h['wind_speed_10m'] for h in hourly_predictions[:24]],
            'predicted_pressure': [h['pressure_msl'] for h in hourly_predictions[:24]],
            'predicted_cloud_cover': [h['cloud_cover'] for h in hourly_predictions[:24]],
            'predicted_visibility': [h['visibility'] for h in hourly_predictions[:24]],
            'predicted_weather_code': [h['weather_code'] for h in hourly_predictions[:24]],
            'predicted_uv_index': [h['uv_index'] for h in hourly_predictions[:24]],
            'hourly_predictions': hourly_predictions[:24],
            'daily_forecast': daily_forecast,
            'prediction_hours': 24
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong predict_storm: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Load model khi import
load_model()