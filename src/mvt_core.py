"""
mvt_core.py
Hàm tìm điểm \"MVT-like\" trên chuỗi thời gian rời rạc.
Ý tưởng: trong mỗi cửa sổ (window) tính slope trung bình giữa start và end,
tìm vị trí có slope cục bộ gần nhất.
"""

import numpy as np
import pandas as pd

def find_mvt_points(series, window=24):
    """
    Input:
      - series: 1-D numpy array or pd.Series (gia tri numeric)
      - window: chiều dài cửa sổ (số bước)
    Output:
      - mvt_indices: numpy array các index (global indices) chỉ vị trí được chọn cho mỗi cửa sổ
    Lưu ý: số lượng trả về = len(series) - window + 1
    """
    x = np.asarray(series)
    n = len(x)
    if n < window:
        return np.array([], dtype=int)
    mvt_indices = []
    for start in range(0, n - window + 1):
        end = start + window - 1
        y_start = x[start]
        y_end = x[end]
        avg_slope = (y_end - y_start) / (end - start)  # slope per-step
        local_slopes = np.diff(x[start:end+1])  # length window-1
        idx_local = int(np.argmin(np.abs(local_slopes - avg_slope)))
        # vị trí index trong series ứng với slope giữa i-1 và i là i
        mvt_index = start + 1 + idx_local
        mvt_indices.append(mvt_index)
    return np.array(mvt_indices, dtype=int)

def mvt_shift_summary(df_orig, df_rounded, feature='temperature', window=24, n_windows=8):
    """
    Trả về DataFrame tóm tắt sự dịch chuyển index MVT giữa gốc và bản làm tròn.
    - df_orig, df_rounded: DataFrame giống nhau index/time
    - feature: tên cột
    - window: cửa sổ
    - n_windows: số cửa sổ đầu để hiển thị
    """
    orig_idx = find_mvt_points(df_orig[feature].values, window=window)
    rnd_idx = find_mvt_points(df_rounded[feature].values, window=window)
    n = min(len(orig_idx), len(rnd_idx), n_windows)
    rows = []
    for i in range(n):
        oi = int(orig_idx[i])
        ri = int(rnd_idx[i])
        rows.append({
            'window_num': i,
            'orig_mvt_index': oi,
            'orig_mvt_time': df_orig.loc[oi, 'time'],
            'orig_mvt_value': float(df_orig.loc[oi, feature]),
            'rounded_mvt_index': ri,
            'rounded_mvt_time': df_rounded.loc[ri, 'time'],
            'rounded_mvt_value': float(df_rounded.loc[ri, feature]),
            'index_shift': ri - oi
        })
    return pd.DataFrame(rows)
