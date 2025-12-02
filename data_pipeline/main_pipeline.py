import sys
import os
import time
from datetime import datetime, timedelta
import requests

# ============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG)
# ============================================================================
# Lấy đường dẫn thư mục hiện tại (.../data_pipeline)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn thư mục gốc dự án (.../weather_project)
project_root = os.path.dirname(current_dir)

# Thêm thư mục gốc vào Python Path để Python hiểu "data_pipeline" là module
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"📂 Project Root: {project_root}")

# ============================================================================
# 2. IMPORT MODULES (ĐÃ SỬA ĐƯỜNG DẪN)
# ============================================================================
try:
    # SỬA LỖI: Bỏ ".data_forecast" vì file nằm trực tiếp trong data_pipeline
    from data_pipeline.data_loader import fetch_weather_api
    from data_pipeline.data_cleaning import clean_api_data
    from data_pipeline.data_storage import connect_to_db, get_last_timestamp, insert_weather_data, get_provinces_from_db
    print("✅ Import modules thành công!")
except ImportError as e:
    print(f"❌ LỖI IMPORT: {e}")
    sys.exit(1)

# ============================================================================
# 3. CẤU HÌNH THAM SỐ CHẠY
# ============================================================================
PROVINCE_DELAY_SECONDS = 0 
YEAR_DELAY_SECONDS = 5
RETRY_DELAY_START = 10
MAX_RETRIES = 5

def process_province_range(conn, province_id, province_name, lat, lon, start_date, end_date):
    print(f"  Đang xử lý khoảng: {start_date} đến {end_date}...")
    retries = 0
    current_delay = RETRY_DELAY_START
    
    while retries < MAX_RETRIES:
        try:
            api_data = fetch_weather_api(lat, lon, start_date, end_date)
            cleaned_df = clean_api_data(api_data, province_id, province_name)
            
            if cleaned_df is not None and not cleaned_df.empty:
                count = insert_weather_data(conn, cleaned_df)
                print(f"  -> Đã lưu {count} dòng dữ liệu.")
            else:
                print(f"  -> Không có dữ liệu hợp lệ.")
            return # Thành công thì thoát luôn
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"  !!! Lỗi 429 (Too Many Requests). Chờ {current_delay}s...")
                time.sleep(current_delay)
                retries += 1
                current_delay *= 2
            else:
                print(f"  !!! Lỗi HTTP khác: {e}")
                break
        except Exception as e:
            print(f"  !!! Lỗi không xác định: {e}")
            break
            
    print(f"  !!! Bỏ qua khoảng {start_date}-{end_date} sau {retries} lần thử.")

def run_pipeline():
    # Kết nối DB
    conn = connect_to_db()
    if not conn:
        return
    
    # Lấy danh sách tỉnh
    provinces = get_provinces_from_db(conn) 
    if not provinces:
        print("❌ Không tìm thấy tỉnh nào trong DB (Bảng 'provinces' trống?).")
        return
        
    print(f"✅ Tìm thấy {len(provinces)} tỉnh cần cập nhật.")
    
    end_date_today_str = datetime.now().strftime('%Y-%m-%d')
    target_year = datetime.now().year

    for province in provinces:
        province_id = province[0]
        name = province[1]
        lat = province[2]
        lon = province[3]
        
        last_ts = get_last_timestamp(conn, province_id) 
        
        print(f"\n==================================================")
        print(f"🌤️  XỬ LÝ: {name} (ID: {province_id})")
        print(f"==================================================")

        if last_ts is None:
            print("  -> Chưa có dữ liệu. Bắt đầu cào từ năm 2020...")
            current_year = 2020
            while current_year <= target_year:
                loop_start_date = f"{current_year}-01-01"
                loop_end_date = f"{current_year}-12-31"
                if current_year == target_year:
                    loop_end_date = end_date_today_str
                
                process_province_range(conn, province_id, name, lat, lon, loop_start_date, loop_end_date)
                
                if current_year < target_year:
                    time.sleep(YEAR_DELAY_SECONDS)
                current_year += 1
        else:
            # Logic cập nhật hàng ngày
            start_date_obj = last_ts.date() + timedelta(days=1)
            start_date_str = start_date_obj.strftime('%Y-%m-%d')
            
            if start_date_str > end_date_today_str:
                print(f"  -> Dữ liệu đã mới nhất ({last_ts}). Bỏ qua.")
                continue
            
            print(f"  -> Cập nhật từ {start_date_str} đến hôm nay...")
            process_province_range(conn, province_id, name, lat, lon, start_date_str, end_date_today_str)

        if PROVINCE_DELAY_SECONDS > 0:
            time.sleep(PROVINCE_DELAY_SECONDS)

    print("\n✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH.")
    conn.close()

if __name__ == "__main__":
    run_pipeline()