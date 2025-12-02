# backend_api/controllers/main_controller.py
# Xử lý các route cho trang Home và News.

from flask import Blueprint, render_template, request, jsonify
from ..models.weather_model import Provinces
from datetime import datetime
import requests

# Tạo một Blueprint
main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/')
def route_home():
    """Phục vụ trang chủ."""
    return render_template('index.html', nav_active='home')

@main_bp.route('/news')
def route_news():
    """Phục vụ trang tin tức."""
    from ..models.news_model import get_latest_news
    news_list = get_latest_news()
    return render_template('news.html', nav_active='news', news_list=news_list)

@main_bp.route('/api/current_weather')
def api_current_weather():
    """API lấy thời tiết hiện tại (trung bình 1 ngày)."""
    province_name = request.args.get('province')  # Từ query
    if not province_name:
        return jsonify({"error": "Thiếu province"}), 400

    province = Provinces.query.filter_by(name=province_name).first()
    if not province:
        return jsonify({"error": "Không tìm thấy tỉnh"}), 404

    try:
        # Gọi Open-Meteo cho current (hourly aggregate to daily avg)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": province.latitude,
            "longitude": province.longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "timezone": "Asia/Bangkok",
            "start_date": datetime.now().strftime('%Y-%m-%d'),
            "end_date": datetime.now().strftime('%Y-%m-%d')
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        hourly = data.get("hourly", {})
        avg_temp = sum(hourly.get("temperature_2m", [])) / len(hourly.get("temperature_2m", [0])) if hourly.get("temperature_2m") else 0
        avg_humidity = sum(hourly.get("relative_humidity_2m", [])) / len(hourly.get("relative_humidity_2m", [0])) if hourly.get("relative_humidity_2m") else 0
        
        return jsonify({
            "province": province_name,
            "avg_temp": round(avg_temp, 1),
            "avg_humidity": round(avg_humidity, 1),
            "precipitation": sum(hourly.get("precipitation", []))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # ... (các code cũ bên trên giữ nguyên)

@main_bp.route('/about')
def about():
    """Trang Về chúng tôi"""
    # nav_active='about' để highlight menu nếu cần
    return render_template('about.html', nav_active='about')

@main_bp.route('/contact')
def contact():
    """Trang Liên hệ"""
    return render_template('contact.html', nav_active='contact')

@main_bp.route('/privacy')
def privacy():
    """Trang Chính sách bảo mật"""
    return render_template('privacy.html')

@main_bp.route('/terms')
def terms():
    """Trang Điều khoản sử dụng"""
    return render_template('terms.html')