# backend_api/controllers/__init__.py


from flask import Flask

def register_blueprints(app: Flask):
    
    # main_bp – luôn có
    from .main_controller import main_bp
    app.register_blueprint(main_bp)
    print("Đã đăng ký: main_bp")

    # forecast_bp – có thể chưa hoàn thiện
    try:
        from .forecast_controller import forecast_bp
        app.register_blueprint(forecast_bp)
        print("Đã đăng ký: forecast_bp")
    except Exception as e:
        print(f"Chưa đăng ký forecast_bp: {e}")

    # chart_bp – CHẮC CHẮN PHẢI CÓ
    try:
        from .chart_controller import chart_bp
        app.register_blueprint(chart_bp)
        print("ĐÃ ĐĂNG KÝ THÀNH CÔNG: chart_bp → /chart hoạt động!")
    except Exception as e:
        print(f"LỖI KHÔNG ĐĂNG KÝ ĐƯỢC chart_bp: {e}")

    