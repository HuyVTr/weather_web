import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Sửa 2 dòng import bên dưới
from .models import db # Thêm dấu "." ở trước
from .controllers import register_blueprints # Thêm dấu "." ở trước

# Thêm sys.path để Python tìm thấy các thư mục con
# Điều này đảm bảo các controller có thể import các model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- CẤU HÌNH DATABASE ---
# Lấy mật khẩu từ biến môi trường (an toàn hơn) hoặc hardcode
# Để an toàn, bạn nên set biến môi trường: set DB_PASSWORD=your_password
DB_USERNAME = "postgres"
DB_PASSWORD = os.environ.get('DB_PASSWORD', '123456')
DB_HOST = "db.rrfonraxfittnyhoekwt.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"

DATABASE_URI = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Cảnh báo nếu chưa đặt mật khẩu
if DB_PASSWORD == 'your_password_here':
    print("*"*50)
    print("CẢNH BÁO: Vui lòng đặt DB_PASSWORD trong file app.py")
    print("*"*50)

def create_app():
    """Tạo và cấu hình Flask app."""
    
    # Tạo app
    # Cấu hình đường dẫn template và static
    app = Flask(__name__, 
                template_folder='templates', 
                static_folder='static')
    
    # Cấu hình database
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Khởi tạo DB với app
    db.init_app(app)

    # Đăng ký các controller (Blueprints)
    register_blueprints(app)

    return app

if __name__ == "__main__":
    app = create_app()
    print("Khởi chạy máy chủ Flask tại http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)