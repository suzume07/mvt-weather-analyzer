"""
preprocess.py
Hàm load/chuẩn hoá dữ liệu thời tiết và các hàm làm tròn (rounding).
Tất cả chú thích bằng tiếng Việt.
"""

import pandas as pd
import numpy as np

def load_weather_data(path, time_col='time'):
    """
    Đọc CSV, parse cột thời gian.
    - path: đường dẫn file CSV
    - time_col: tên cột thời gian (mặc định 'time')
    Trả về DataFrame có cột time kiểu datetime.
    """
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def ensure_time_index(df, time_col='time'):
    """Chắc chắn cột time là datetime và index tăng dần."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def fill_missing(df, method='ffill', cols=None):
    """
    Impute missing:
    - method: 'ffill' hoặc 'mean' hoặc 'drop'
    - cols: danh sách cột cần xử lý (None => xử lý toàn bộ)
    """
    df = df.copy()
    if cols is None:
        cols = df.columns.tolist()
    if method == 'ffill':
        df[cols] = df[cols].fillna(method='ffill').fillna(method='bfill')
    elif method == 'mean':
        for c in cols:
            if df[c].dtype.kind in 'biufc':
                df[c] = df[c].fillna(df[c].mean())
    elif method == 'drop':
        df = df.dropna(subset=cols)
    return df

def cardinal_to_degrees(s):
    """Chuyển hướng gió dạng 'N', 'NE'... sang degrees (nếu cần)."""
    # mapping cơ bản (có thể mở rộng)
    mapping = {
        'N': 0, 'NNE': 22.5, 'NE':45, 'ENE':67.5,
        'E':90,'ESE':112.5,'SE':135,'SSE':157.5,
        'S':180,'SSW':202.5,'SW':225,'WSW':247.5,
        'W':270,'WNW':292.5,'NW':315,'NNW':337.5
    }
    return s.map(mapping)

def round_decimals(series, decimals):
    """Làm tròn series theo số chữ số thập phân (decimals)."""
    return series.round(decimals)

def round_to_step(series, step):
    """Làm tròn series tới nearest 'step' (ví dụ step=5 => nearest 5 units)."""
    return (np.round(series / step) * step)

def apply_rounding_single(df, feature, scheme):
    """
    Áp rounding cho 1 cột.
    - scheme: {'type':'decimals'|'step', 'levels':[...]}
    Trả về list các tuple (level, df_rounded).
    """
    results = []
    for level in scheme.get('levels', []):
        df_r = df.copy()
        if scheme['type'] == 'decimals':
            if level >= 0:
                df_r[feature] = round_decimals(df_r[feature], level)
            else:
                # negative decimals -> ví dụ level=-1 => làm tròn tens
                factor = 10 ** (-level)
                df_r[feature] = np.round(df_r[feature] / factor) * factor
        elif scheme['type'] == 'step':
            df_r[feature] = round_to_step(df_r[feature], level)
        else:
            raise ValueError("Unknown rounding type")
        results.append((level, df_r))
    return results

def apply_rounding_multi(df, rounding_schemes):
    """
    Áp rounding cho nhiều cột theo schemes.
    - rounding_schemes: dict, key=feature, value=scheme dict (giống ở trên)
    Trả về list các dict: [{'feature':f,'level':lvl,'df':df_rounded}, ...]
    """
    outs = []
    for feature, scheme in rounding_schemes.items():
        outs.extend([{'feature': feature, 'level': lvl, 'df': dfr} for (lvl, dfr) in apply_rounding_single(df, feature, scheme)])
    return outs
