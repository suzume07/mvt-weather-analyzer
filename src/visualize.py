"""
visualize.py
Hàm vẽ matplotlib để so sánh chuỗi gốc vs rounded và biểu đồ error.
"""

import matplotlib.pyplot as plt
import os

def plot_temp_rounded(df_orig, df_rounded, days=7, save_path=None):
    """
    Vẽ chuỗi nhiệt độ gốc và bản làm tròn trong 'days' ngày đầu.
    Nếu save_path được truyền, lưu ảnh vào đó (định dạng png).
    """
    n = min(len(df_orig), days * 24)
    plt.figure(figsize=(10,4))
    plt.plot(df_orig['time'][:n], df_orig['temperature'][:n], label='original')
    plt.plot(df_rounded['time'][:n], df_rounded['temperature'][:n], label='rounded')
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.title(f'Temperature: original vs rounded ({days} days)')
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_error_vs_rounding(results_df, feature='temperature', save_path=None):
    """
    results_df kỳ vọng có các cột: ['feature','level','mse','mae','r2']
    """
    sub = results_df[results_df['feature'] == feature]
    if sub.empty:
        print("Không có dữ liệu để vẽ cho feature:", feature)
        return
    x = list(sub['level'].astype(str))
    y = list(sub['mse'])
    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker='o')
    plt.title(f'Effect of rounding on {feature} (MSE)')
    plt.xlabel('rounding level')
    plt.ylabel('MSE')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_mvt_points(df, mvt_indices, feature='temperature', save_path=None):
    """
    Vẽ chuỗi và đánh dấu mvt_indices (list/array các index)
    """
    plt.figure(figsize=(10,4))
    plt.plot(df['time'], df[feature], label=feature)
    if len(mvt_indices) > 0:
        plt.scatter(df.loc[mvt_indices, 'time'], df.loc[mvt_indices, feature], c='red', s=20, label='MVT points')
    plt.xlabel('time')
    plt.ylabel(feature)
    plt.title(f'{feature} with MVT points')
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
