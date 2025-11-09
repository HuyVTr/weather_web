# backend_api/controllers/storm_controller.py
# Xử lý các route cho trang Bão và API Bão.

from flask import Blueprint, render_template, jsonify
import pandas as pd
import sys, os
import numpy as np
from sklearn.cluster import DBSCAN
from ..models.storm_model import get_storm_warning_level
from datetime import timedelta  # Thêm import
import requests  # Thêm import

# Thêm thư mục gốc vào path để import predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from machine_learning.predictor import predict_storm
from ..models.weather_model import Provinces

storm_bp = Blueprint('storm_bp', __name__)

@storm_bp.route('/storm')
def route_storm():
    """Phục vụ trang dự báo bão."""
    return render_template('storm.html', nav_active='storm')

@storm_bp.route('/api/storm_track')
def api_get_storm_track():
    """
    API dự đoán đường đi của bão.
    Dùng anomaly detection trên Biển Đông + ML predict track.
    """
    # Fetch grid data Biển Đông (5x5 points)
    lats = np.arange(10, 20, 2)
    lons = np.arange(108, 116, 2)
    anomalies = []
    for lat in lats:
        for lon in lons:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=pressure_msl,wind_speed_10m&forecast_days=3"
            data = requests.get(url).json()
            hourly_p = data['hourly']['pressure_msl'][:24]  # Next 24h
            hourly_w = data['hourly']['wind_speed_10m'][:24]
            if any(p < 980 and w > 33 for p, w in zip(hourly_p, hourly_w)):  # Điều chỉnh ngưỡng
                anomalies.append({'lat': lat, 'lon': lon, 'intensity': min(hourly_p), 'max_wind': max(hourly_w)})

    if len(anomalies) < 3:
        return jsonify({"no_storm": True, "message": "Không phát hiện bão"})

    # Cluster để tìm center
    coords = np.array([[a['lat'], a['lon']] for a in anomalies])
    clustering = DBSCAN(eps=2, min_samples=3).fit(coords)  # Cluster radius 2°
    if max(clustering.labels_) < 0:  # No cluster
        return jsonify(get_mock_storm_track())

    # Tìm center từ cluster lớn nhất
    cluster_labels = clustering.labels_
    center_indices = np.where(cluster_labels >= 0)[0]
    center_lat, center_lon = coords[center_indices].mean(axis=0)
    max_wind = max(a['max_wind'] for a in anomalies if a['lat'] in center_lat and a['lon'] in center_lon)

    # Predict track using ML
    input_df = pd.DataFrame([[25, min(hourly_p), max_wind]], columns=['temp_lag1', 'pressure_lag1', 'wind_lag1'])
    pred = predict_storm(input_df)
    if 'error' in pred:
        return jsonify({"error": pred['error']})

    # Tạo track dựa trên dự đoán (giả sử pred trả về nhiệt độ 24h, dùng để điều chỉnh hướng)
    pred_temps = pred.get('predicted_temperature', [])
    track = [[center_lon, center_lat]]
    for i in range(min(3, len(pred_temps))):  # Dự đoán 3 điểm
        # Điều chỉnh hướng dựa trên thay đổi nhiệt độ (giả sử nhiệt độ giảm khi bão di chuyển)
        delta_lon = -2 * (1 - pred_temps[i] / 25) if pred_temps[i] < 25 else -1  # Di chuyển về tây
        delta_lat = 1 if i % 2 == 0 else 0  # Di chuyển lên bắc nhẹ
        track.append([track[-1][0] + delta_lon, track[-1][1] + delta_lat])

    # Kiểm tra landfall (tối ưu hóa bằng cách lấy danh sách tọa độ tỉnh trước)
    province_coords = [(p.latitude, p.longitude) for p in Provinces.query.all()]
    landfall = any(abs(lat - p[0]) < 0.5 and abs(lon - p[1]) < 0.5 for lat, lon in track[1:] for p in province_coords)

    return jsonify({
        "storm_detected": True,
        "center": [center_lat, center_lon],
        "track": {"type": "LineString", "coordinates": track},
        "landfall_vn": landfall,
        "warning": get_storm_warning_level(max_wind)
    })

def get_mock_storm_track():
    """Tạo dữ liệu bão giả."""
    return {
        "type": "FeatureCollection",
        "features": [
            { "type": "Feature", "geometry": { "type": "LineString", "coordinates": [[118.0, 12.0], [117.0, 12.5], [116.0, 13.0], [114.5, 14.0]] }},
            { "type": "Feature", "geometry": { "type": "Point", "coordinates": [114.5, 14.0] }, "properties": { "name": "Tâm bão (t+0)" } }
        ]
    }