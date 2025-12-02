# backend_api/controllers/chart_controller.py
from flask import Blueprint, render_template

chart_bp = Blueprint('chart_bp', __name__, template_folder='../templates')

@chart_bp.route('/chart')
def route_chart():
    return render_template('chart.html', nav_active='chart')