# backend_api/controllers/forecast_controller.py
# Xử lý các route cho trang Dự báo và API thời tiết.

from flask import Blueprint, render_template, request, jsonify
from ..models.weather_model import Provinces
from data_pipeline.data_storage import connect_to_db, get_last_timestamp
from machine_learning.model_training import load_data_for_training, feature_engineering, train_model, save_model
from machine_learning.predictor import load_model, predict_storm
import requests
import pandas as pd
from datetime import datetime, timedelta

forecast_bp = Blueprint('forecast_bp', __name__)

@forecast_bp.route('/forecast')
def route_forecast():
    """Phục vụ trang dự báo."""
    return render_template('forecast.html', nav_active='forecast')

@forecast_bp.route('/api/provinces')
def api_get_provinces():
    """API lấy danh sách 63 tỉnh."""
    try:
        provinces = Provinces.query.order_by(Provinces.name).all()
        return jsonify([p.to_dict() for p in provinces])
    except Exception as e:
        print(f"Lỗi /api/provinces: {e}")
        return jsonify({"error": "Không thể lấy danh sách tỉnh"}), 500

@forecast_bp.route('/api/forecast')
def api_get_forecast():
    """API lấy dữ liệu thời tiết (từ Open-Meteo + ML nếu cần + AQI)."""
    province_name = request.args.get('province', '')  # Từ query hoặc geolocation
    if not province_name:
        return jsonify({"error": "Thiếu province"}), 400

    province = Provinces.query.filter_by(name=province_name).first()
    if not province:
        return jsonify({"error": "Không tìm thấy tỉnh"}), 404

    # Check data sufficiency
    conn = connect_to_db()
    last_ts = get_last_timestamp(conn, province.province_id)
    data_hours = 24 if last_ts and (datetime.now() - last_ts).total_seconds() / 3600 < 24 else 0
    use_ml = data_hours < 24  # Nếu <24h, dùng ML predict
    conn.close()

    try:
        # Fetch weather from Open-Meteo
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": province.latitude,
            "longitude": province.longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,showers,weather_code,pressure_msl,wind_speed_10m,wind_direction_10m,visibility,uv_index",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl,visibility,uv_index",
            "timezone": "Asia/Bangkok",
            "forecast_days": 7
        }
        response = requests.get(url, params=params)
        data = response.json()

        forecast_data = {
            "location": province_name,
            "current": data.get("current", {}),
            "daily": data.get("daily", {}),
            "hourly": {k: v[:48] for k, v in data.get("hourly", {}).items()}  # 48h
        }

        # Fetch AQI from WAQI (demo token)
        aqi_url = f"https://api.waqi.info/feed/geo:{province.latitude};{province.longitude}/?token=demo"
        aqi_response = requests.get(aqi_url)
        if aqi_response.status_code == 200:
            aqi_data = aqi_response.json().get('data', {})
            forecast_data['aqi'] = {
                'index': aqi_data.get('aqi', 0),
                'components': aqi_data.get('iaqi', {})
            }
        else:
            forecast_data['aqi'] = {'index': 0, 'components': {}}

        if use_ml:
            # Retrain nếu insufficient
            df = load_data_for_training()
            if not df.empty and len(df) < 24:
                start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                query = f"""
                    SELECT "timestamp", temperature_2m, pressure_msl, wind_speed_10m
                    FROM weather_data
                    WHERE province_id = {province.province_id} AND timestamp >= '{start_date}'
                    ORDER BY "timestamp" DESC
                    LIMIT 50000
                """
                df = pd.read_sql(query, connect_to_db())
            if not df.empty:
                X, y = feature_engineering(df)
                if not X.empty:
                    model, _, _ = train_model(X, y)
                    save_model(model)
            
            # Predict hourly/7day
            input_df = pd.DataFrame([[forecast_data['current'].get('temperature_2m', 25), 
                                    forecast_data['current'].get('pressure_msl', 1013), 
                                    forecast_data['current'].get('wind_speed_10m', 5)]], 
                                    columns=['temp_lag1', 'pressure_lag1', 'wind_lag1'])
            ml_pred = predict_storm(input_df)  # Cần chỉnh predictor.py để trả về mảng 24h
            forecast_data['ml_prediction'] = ml_pred if 'predicted_temperature' in ml_pred else {}

        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({"error": f"Lỗi khi lấy dữ liệu: {e}"}), 500