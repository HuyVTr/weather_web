# services/forecast_ml/cron_job.py
import time
import json
import sys
import os
from datetime import datetime
from sqlalchemy import create_engine, text

# --- 1. CẤU HÌNH ĐƯỜNG DẪN (Để import được các module khác) ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # services/forecast_ml
services_dir = os.path.dirname(current_dir)              # services
project_root = os.path.dirname(services_dir)             # WEATHER_PROJECT
sys.path.append(project_root)

# Import hàm dự báo từ project
try:
    from services.forecast_ml.predictor import predict_storm
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("Vui lòng đảm bảo bạn đang chạy file từ thư mục gốc của dự án hoặc cấu trúc thư mục đúng.")
    sys.exit(1)

# --- 2. CẤU HÌNH DATABASE (Theo yêu cầu của bạn) ---
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "weather_project", # Tên database
    "user": "postgres",          # Username
    "password": "123456"         # Mật khẩu
}

# Tạo chuỗi kết nối cho SQLAlchemy
DB_URI = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

try:
    engine = create_engine(DB_URI)
    print(f"✅ Đã kết nối tới Database: {DB_CONFIG['dbname']}")
except Exception as e:
    print(f"❌ Lỗi cấu hình Database: {e}")
    sys.exit(1)

# --- 3. HÀM CẬP NHẬT DỰ BÁO ---
def update_all_forecasts():
    print(f"\n🚀 [CRON] Bắt đầu cập nhật dự báo lúc {datetime.now()}")
    
    # Lấy danh sách ID tỉnh từ Database
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT province_id, name FROM provinces ORDER BY province_id"))
            provinces = result.fetchall()
        except Exception as e:
            print(f"❌ Lỗi truy vấn danh sách tỉnh: {e}")
            return

    # Tính toán cho từng tỉnh
    count_success = 0
    for p_id, p_name in provinces:
        try:
            print(f"   ⏳ Đang tính toán: {p_name}...", end="", flush=True)
            
            # Gọi hàm dự báo AI
            ml_result = predict_storm(province_id=p_id)
            
            if "error" in ml_result:
                print(f" ⚠️ Lỗi model: {ml_result['error']}")
                continue

            # Chuẩn bị dữ liệu JSON
            json_data = json.dumps(ml_result)
            
            # Lưu vào bảng weather_forecast_cache (UPSERT)
            query = text("""
                INSERT INTO weather_forecast_cache (province_id, updated_at, forecast_data)
                VALUES (:pid, NOW(), :data)
                ON CONFLICT (province_id) 
                DO UPDATE SET 
                    updated_at = NOW(),
                    forecast_data = :data;
            """)
            
            with engine.begin() as conn:
                conn.execute(query, {"pid": p_id, "data": json_data})
                
            print(" ✅ Đã lưu.")
            count_success += 1
            
        except Exception as e:
            print(f" ❌ Lỗi ngoại lệ: {e}")

    print(f"🏁 [CRON] Hoàn tất! Cập nhật thành công {count_success}/{len(provinces)} tỉnh.")

if __name__ == "__main__":
    print(f"🤖 Worker đang chạy...")
    print(f"   Database: {DB_CONFIG['dbname']}")
    print("   (Nhấn Ctrl+C để dừng)")
    
    # Chạy ngay lần đầu tiên khi khởi động
    update_all_forecasts()
    
    while True:
        print("💤 Ngủ 60 phút...")
        time.sleep(3600) # Chạy lại sau 1 tiếng