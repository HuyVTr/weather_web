import psycopg2
import psycopg2.extras as extras
import pandas as pd
from io import StringIO
import traceback

# --- CẤU HÌNH DATABASE ---
# Cần được thay đổi cho phù hợp
DB_CONFIG = {
    "host": "db.rrfonraxfittnyhoekwt.supabase.co",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "123456" 
}

def connect_to_db():
    """
    Kết nối đến PostgreSQL database và trả về connection object.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Kết nối database thành công.")
        return conn
    except Exception as e:
        print(f"LỖI KẾT NỐI DATABASE: {e}\n\nVui lòng kiểm tra lại DB_CONFIG và đảm bảo PostgreSQL đang chạy.")
        return None

def get_provinces_from_db(conn):
    """
    Lấy danh sách các tỉnh (ID, Tên, Lat, Lon) từ bảng provinces.
    """
    provinces = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT province_id, name, latitude, longitude FROM provinces")
            provinces = cur.fetchall()
    except Exception as e:
        print(f"LỖI khi lấy danh sách tỉnh: {e}")
        conn.rollback() # Hoàn tác nếu có lỗi
    return provinces

# --- HÀM MỚI ---
def get_last_timestamp(conn, province_id):
    """
    Lấy timestamp mới nhất của một tỉnh đã có trong database.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT MAX("timestamp") FROM weather_data WHERE province_id = %s',
                (province_id,)
            )
            # fetchone() trả về (datetime_object,) hoặc (None,)
            result = cur.fetchone()[0] 
            return result  # Sẽ là None nếu tỉnh này chưa có dữ liệu
    except Exception as e:
        print(f"  Lỗi khi lấy timestamp cuối cùng: {e}")
        conn.rollback()
        return None
# --- KẾT THÚC HÀM MỚI ---

def insert_weather_data(conn, df):
    """
    Chèn (hoặc cập nhật) dữ liệu thời tiết từ DataFrame vào database
    sử dụng ON CONFLICT DO UPDATE.
    """
    if df.empty:
        return 0
        
    # Tạo một buffer (bộ đệm)
    buffer = StringIO()
    # Ghi DataFrame vào buffer dưới dạng CSV (không có header, index)
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0) # Đưa con trỏ về đầu buffer
    
    # Lấy danh sách cột từ DataFrame (và đảm bảo "timestamp" có dấu ngoặc kép)
    cols = list(df.columns)
    cols[cols.index('timestamp')] = '"timestamp"'
    cols_sql = ', '.join(cols)
    
    # Tạo danh sách các cột để UPDATE
    # Ví dụ: temperature_2m = EXCLUDED.temperature_2m, ...
    update_cols = ", ".join([f'{col} = EXCLUDED.{col}' for col in cols if col not in ['province_id', '"timestamp"']])

    sql_insert = f"""
    INSERT INTO weather_data ({cols_sql})
    VALUES %s
    ON CONFLICT (province_id, "timestamp") DO UPDATE SET
        {update_cols}
    """
    
    cursor = None
    try:
        cursor = conn.cursor()
        # Dùng execute_values để chèn hiệu quả từ buffer
        extras.execute_values(
            cursor,
            sql_insert,
            df.values,
            page_size=1000 # Chèn mỗi lần 1000 dòng
        )
        conn.commit()
        return len(df)
    except Exception as e:
        print(f"LỖI khi chèn dữ liệu: {e}")
        conn.rollback()
        traceback.print_exc() # In ra traceback chi tiết
        return 0
    finally:
        if cursor:
            cursor.close()
