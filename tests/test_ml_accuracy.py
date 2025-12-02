import os
import sys
import argparse
import requests
import numpy as np

# Thêm đường dẫn gốc project để Python tìm thấy các package con
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import các modules nội bộ
from services.forecast_ml.predictor import predict_storm
from backend_api.models.weather_model import Provinces

# SỬA LỖI IMPORTERROR
# Import db và hàm create_app (Application Factory)
from app import db, create_app 

# Khởi tạo instance Flask (đã cấu hình DB)
flask_app = create_app()


def compare_ml_vs_api(province_name="Hà Nội"):
    """So sánh dự đoán ML với API thực tế"""
    
    with flask_app.app_context(): # Sử dụng flask_app đã tạo
        province = Provinces.query.filter_by(name=province_name).first()
        
        if not province:
            print(f"Không tìm thấy tỉnh {province_name}")
            return None
        
        print(f"\n{'='*80}")
        print(f"SO SÁNH DỰ ĐOÁN ML vs API THỰC TẾ - {province_name.upper()}")
        print(f"{'='*80}\n")
        
        # Dự đoán từ ML
        print("Đang chạy ML prediction...")
        ml_result = predict_storm(province.province_id)
        
        if 'error' in ml_result:
            print(f"Lỗi ML: {ml_result['error']}")
            return None
        
        print("ML prediction hoàn thành!")
        
        # Lấy dữ liệu thực từ API
        print("Đang lấy dữ liệu từ Open-Meteo API...")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": province.latitude,
            "longitude": province.longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl,visibility",
            "timezone": "Asia/Bangkok",
            "forecast_days": 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            api_data = response.json()
            print("API data hoàn thành!")
        except Exception as e:
            print(f"Lỗi khi lấy API: {e}")
            return None
        
        # So sánh 24 giờ tới
        ml_temps = ml_result['predicted_temperature'][:24]
        ml_humidity = ml_result['predicted_humidity'][:24]
        ml_precip = ml_result['predicted_precipitation'][:24]
        ml_wind = ml_result['predicted_wind_speed'][:24]
        ml_pressure = ml_result['predicted_pressure'][:24]
        ml_visibility = ml_result['predicted_visibility'][:24]
        
        api_temps = api_data['hourly']['temperature_2m'][:24]
        api_humidity = api_data['hourly']['relative_humidity_2m'][:24]
        api_precip = api_data['hourly']['precipitation'][:24]
        api_wind = api_data['hourly']['wind_speed_10m'][:24]
        api_pressure = api_data['hourly']['pressure_msl'][:24]
        api_visibility = api_data['hourly']['visibility'][:24]
        
        # Tính sai số
        temp_mae = np.mean(np.abs(np.array(ml_temps) - np.array(api_temps)))
        humidity_mae = np.mean(np.abs(np.array(ml_humidity) - np.array(api_humidity)))
        precip_mae = np.mean(np.abs(np.array(ml_precip) - np.array(api_precip)))
        wind_mae = np.mean(np.abs(np.array(ml_wind) - np.array(api_wind)))
        pressure_mae = np.mean(np.abs(np.array(ml_pressure) - np.array(api_pressure)))
        visibility_mae = np.mean(np.abs(np.array(ml_visibility) - np.array(api_visibility))) / 1000
        
        print(f"\nSAI SỐ TRUNG BÌNH (MAE) so với API:")
        print(f"   • Nhiệt độ:     {temp_mae:.2f}°C")
        print(f"   • Độ ẩm:        {humidity_mae:.2f}%")
        print(f"   • Lượng mưa:    {precip_mae:.2f}mm")
        print(f"   • Tốc độ gió:   {wind_mae:.2f}km/h")
        print(f"   • Áp suất:      {pressure_mae:.2f}hPa")
        print(f"   • Tầm nhìn:     {visibility_mae:.2f}km")
        
        # Tính RMSE + R²
        temp_rmse = np.sqrt(np.mean((np.array(ml_temps) - np.array(api_temps))**2))
        humidity_rmse = np.sqrt(np.mean((np.array(ml_humidity) - np.array(api_humidity))**2))
        
        def r2_score(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        temp_r2 = r2_score(np.array(api_temps), np.array(ml_temps))
        humidity_r2 = r2_score(np.array(api_humidity), np.array(ml_humidity))
        
        print(f"\nSAI SỐ RMSE:")
        print(f"   • Nhiệt độ:     {temp_rmse:.2f}°C")
        print(f"   • Độ ẩm:        {humidity_rmse:.2f}%")
        print(f"\nHỆ SỐ R²:")
        print(f"   • Nhiệt độ:     {temp_r2:.3f}")
        print(f"   • Độ ẩm:        {humidity_r2:.3f}")
        
        # Đánh giá chi tiết
        scores = []
        # Nhiệt độ
        if temp_mae < 1.5 and temp_r2 > 0.90:
            print("   Nhiệt độ: XUẤT SẮC"), scores.append(3)
        elif temp_mae < 2.5 and temp_r2 > 0.80:
            print("   Nhiệt độ: TỐT"), scores.append(2)
        elif temp_mae < 3.5:
            print("   Nhiệt độ: CHẤP NHẬN ĐƯỢC"), scores.append(1)
        else:
            print("   Nhiệt độ: CẦN CẢI THIỆN"), scores.append(0)
            
        # Độ ẩm
        if humidity_mae < 4 and humidity_r2 > 0.85:
            print("   Độ ẩm: XUẤT SẮC"), scores.append(3)
        elif humidity_mae < 7 and humidity_r2 > 0.75:
            print("   Độ ẩm: TỐT"), scores.append(2)
        elif humidity_mae < 10:
            print("   Độ ẩm: CHẤP NHẬN ĐƯỢC"), scores.append(1)
        else:
            print("   Độ ẩm: CẦN CẢI THIỆN"), scores.append(0)
            
        # Lượng mưa & gió
        scores.append(2 if precip_mae < 1.5 else 1 if precip_mae < 3 else 0)
        scores.append(2 if wind_mae < 2 else 1 if wind_mae < 4 else 0)
        
        avg_score = np.mean(scores)
        print(f"\n{'='*80}")
        print(f"TỔNG KẾT: Điểm trung bình: {avg_score:.2f}/3.0")
        if avg_score >= 2.5:
            print("   MÔ HÌNH CÓ ĐỘ TIN CẬY RẤT CAO! Có thể dùng thực tế")
        elif avg_score >= 1.5:
            print("   MÔ HÌNH TỐT - Phù hợp triển khai")
        elif avg_score >= 1.0:
            print("   MÔ HÌNH CHẤP NHẬN ĐƯỢC - Nên train thêm")
        else:
            print("   MÔ HÌNH CẦN CẢI THIỆN")
        print(f"{'='*80}")
        
        # In 10 giờ đầu
        print(f"\nCHI TIẾT 10 GIỜ ĐẦU TIÊN:")
        print(f"{'Giờ':<6} {'ML Temp':<10} {'API Temp':<10} {'Diff':<8} {'ML Hum':<8} {'API Hum':<8}")
        print("-" * 60)
        for i in range(min(10, len(ml_temps))):
            t_diff = ml_temps[i] - api_temps[i]
            h_diff = ml_humidity[i] - api_humidity[i]
            print(f"{i+1:<6} {ml_temps[i]:<10.1f} {api_temps[i]:<10.1f} {t_diff:+.1f}     {ml_humidity[i]:<8.0f} {api_humidity[i]:<8.0f}")
        print(f"\n{'='*80}\n")
        
        return {
            'temp_mae': temp_mae, 'humidity_mae': humidity_mae,
            'precip_mae': precip_mae, 'wind_mae': wind_mae,
            'temp_r2': temp_r2, 'humidity_r2': humidity_r2,
            'avg_score': avg_score
        }


def test_multiple_provinces():
    provinces = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ"]
    results = [compare_ml_vs_api(p) for p in provinces if compare_ml_vs_api(p)]
    if results:
        overall = np.mean([r['avg_score'] for r in results])
        print(f"\nĐIỂM TRUNG BÌNH TẤT CẢ TỈNH: {overall:.2f}/3.0")
        print("MÔ HÌNH RẤT TỐT!" if overall >= 2.5 else "MÔ HÌNH TỐT" if overall >= 1.5 else "CẦN CẢI THIỆN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--province', type=str, help='Tên tỉnh')
    parser.add_argument('--all', action='store_true', help='Test tất cả tỉnh')
    args = parser.parse_args()
    
    if args.all:
        test_multiple_provinces()
    elif args.province:
        compare_ml_vs_api(args.province)
    else:
        compare_ml_vs_api("Hà Nội")