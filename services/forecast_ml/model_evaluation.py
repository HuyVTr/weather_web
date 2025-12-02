# machine_learning/model_evaluation.py
# Trách nhiệm: Tải model và đánh giá trên tập test.

import joblib
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_learning.model_training import (
    load_data_for_training, 
    feature_engineering, 
    MODEL_PATH
)

def evaluate():
    """Đánh giá mô hình đã train."""
    print("\n" + "="*60)
    print("BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH")
    print("="*60 + "\n")
    
    # 1. Kiểm tra model có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy mô hình tại {MODEL_PATH}")
        print("Vui lòng chạy model_training.py trước để tạo mô hình.")
        return False
    
    # 2. Tải mô hình
    print(f"📦 Đang tải mô hình từ {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print("✓ Tải mô hình thành công!\n")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return False
    
    # 3. Tải và xử lý dữ liệu
    print("📊 Đang tải dữ liệu để đánh giá...")
    df = load_data_for_training()
    
    if df.empty:
        print("❌ Không có dữ liệu để đánh giá")
        return False
    
    X, y = feature_engineering(df)
    
    if X.empty:
        print("❌ Không có features để đánh giá")
        return False
    
    print(f"✓ Đã tải {len(X)} mẫu dữ liệu\n")
    
    # 4. Tái tạo train/test split (dùng cùng random_state)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📈 Kích thước tập train: {len(X_train)}")
    print(f"📉 Kích thước tập test: {len(X_test)}\n")
    
    # 5. Dự đoán
    print("🔮 Đang thực hiện dự đoán...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 6. Tính các metrics
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*60)
    
    # Train metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print("\n📊 HIỆU SUẤT TRÊN TẬP TRAIN:")
    print(f"   • MSE (Mean Squared Error):  {train_mse:.4f}")
    print(f"   • RMSE (Root MSE):           {train_rmse:.4f}°C")
    print(f"   • MAE (Mean Absolute Error): {train_mae:.4f}°C")
    print(f"   • R² Score:                  {train_r2:.4f}")
    
    # Test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n📊 HIỆU SUẤT TRÊN TẬP TEST:")
    print(f"   • MSE (Mean Squared Error):  {test_mse:.4f}")
    print(f"   • RMSE (Root MSE):           {test_rmse:.4f}°C")
    print(f"   • MAE (Mean Absolute Error): {test_mae:.4f}°C")
    print(f"   • R² Score:                  {test_r2:.4f}")
    
    # 7. Phân tích overfitting/underfitting
    print("\n📈 PHÂN TÍCH:")
    diff_rmse = abs(train_rmse - test_rmse)
    diff_r2 = abs(train_r2 - test_r2)
    
    if diff_rmse < 0.5 and diff_r2 < 0.05:
        print("   ✓ Mô hình cân bằng tốt (không overfitting)")
    elif train_rmse < test_rmse and train_r2 > test_r2:
        print("   ⚠ Có dấu hiệu overfitting nhẹ")
    else:
        print("   ℹ Mô hình hoạt động ổn định")
    
    if test_r2 > 0.8:
        print("   ✓ Độ chính xác cao (R² > 0.8)")
    elif test_r2 > 0.6:
        print("   ℹ Độ chính xác khá (R² > 0.6)")
    else:
        print("   ⚠ Độ chính xác cần cải thiện (R² < 0.6)")
    
    # 8. Visualize nếu có matplotlib
    try:
        print("\n📊 Đang tạo biểu đồ...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted (Train)
        axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_train.min(), y_train.max()], 
                        [y_train.min(), y_train.max()], 
                        'r--', lw=2)
        axes[0, 0].set_xlabel('Thực tế (°C)')
        axes[0, 0].set_ylabel('Dự đoán (°C)')
        axes[0, 0].set_title(f'Train Set: R²={train_r2:.4f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Actual vs Predicted (Test)
        axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=10, color='green')
        axes[0, 1].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', lw=2)
        axes[0, 1].set_xlabel('Thực tế (°C)')
        axes[0, 1].set_ylabel('Dự đoán (°C)')
        axes[0, 1].set_title(f'Test Set: R²={test_r2:.4f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals (Train)
        residuals_train = y_train.values.ravel() - y_train_pred
        axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.5, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Dự đoán (°C)')
        axes[1, 0].set_ylabel('Residuals (°C)')
        axes[1, 0].set_title(f'Train Residuals: MAE={train_mae:.4f}°C')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals (Test)
        residuals_test = y_test.values.ravel() - y_test_pred
        axes[1, 1].scatter(y_test_pred, residuals_test, alpha=0.5, s=10, color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Dự đoán (°C)')
        axes[1, 1].set_ylabel('Residuals (°C)')
        axes[1, 1].set_title(f'Test Residuals: MAE={test_mae:.4f}°C')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_dir = os.path.join(os.path.dirname(__file__), 'evaluation_results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'model_evaluation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Đã lưu biểu đồ tại: {output_path}")
        
        # Hiển thị biểu đồ
        # plt.show()  # Uncomment nếu muốn hiển thị
        
    except Exception as e:
        print(f"⚠ Không thể tạo biểu đồ: {e}")
    
    print("\n" + "="*60)
    print("HOÀN TẤT ĐÁNH GIÁ")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    evaluate()