"""
run_experiment.py
Script chạy toàn bộ pipeline:
- nếu mode=synth: tạo dữ liệu giả vào data/raw_weather.csv
- load dữ liệu
- áp rounding cho các cột theo scheme
- đánh giá model baseline cho từng bản rounded
- lưu kết quả (CSV) và hình ảnh (outputs/)
"""

import os
import argparse
import pandas as pd

from src.generate_synthetic_weather import generate_synthetic_weather
from src.preprocess import load_weather_data, apply_rounding_multi, ensure_time_index
from src.model_eval import evaluate
from src.mvt_core import find_mvt_points, mvt_shift_summary
from src.visualize import plot_temp_rounded, plot_error_vs_rounding, plot_mvt_points

def main(args):
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    # 1) dữ liệu
    if args.mode == 'synth':
        print("Tạo dữ liệu giả...")
        df = generate_synthetic_weather(hours=args.hours, start=args.start)
        raw_path = os.path.join('data', 'raw_weather.csv')
        df.to_csv(raw_path, index=False)
        print("Saved synthetic data to", raw_path)
    else:
        raw_path = args.data
        if raw_path is None:
            raise ValueError("Cần cung cấp --data PATH nếu mode != synth")
        df = load_weather_data(raw_path)

    df = ensure_time_index(df, time_col='time')

    # 2) cấu hình rounding schemes (mặc định)
    rounding_schemes = {
        'temperature': {'type':'decimals','levels':[2,1,0,-1]},  # -1 => tens
        'humidity': {'type':'step','levels':[1,5,10]}
    }

    # 3) tạo các bản rounded
    variants = apply_rounding_multi(df, rounding_schemes)

    # 4) evaluate cho mỗi variant
    records = []
    for v in variants:
        feat = v['feature']
        lvl = v['level']
        df_r = v['df']
        # lưu file rounded
        fname = f"data/rounded_{feat}_{str(lvl).replace('.', '_')}.csv"
        df_r.to_csv(fname, index=False)
        # evaluate model (dự đoán temperature luôn dù feature làm tròn có thể là khác)
        metrics, model, test_info = evaluate(df_r, feature='temperature')
        records.append({
            'feature': feat,
            'level': lvl,
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'rounded_file': fname
        })
        print(f"Evaluated {feat} level={lvl}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

    results_df = pd.DataFrame(records)
    results_df.to_csv('data/metrics_rounding.csv', index=False)
    print("Saved metrics to data/metrics_rounding.csv")

    # 5) vẽ 1 ví dụ: original vs rounded temp (chọn bản temperature level=0 nếu có)
    temp0 = results_df[(results_df['feature']=='temperature') & (results_df['level']==0)]
    if not temp0.empty:
        # load corresponding rounded file
        df_r0 = pd.read_csv(temp0.iloc[0]['rounded_file'], parse_dates=['time'])
        plot_temp_rounded(df, df_r0, days=7, save_path='outputs/figures/temp_orig_vs_rounded0.png')
    # 6) plot error vs rounding for temperature
    plot_error_vs_rounding(results_df, feature='temperature', save_path='outputs/figures/error_vs_rounding_temperature.png')

    # 7) mvt shift example (first 8 windows)
    # chọn rounded 0 decimals nếu tồn tại, nếu không lấy bản đầu tiên
    chosen = df_r0 if not temp0.empty else variants[0]['df']
    mvt_summary = mvt_shift_summary(df, chosen, feature='temperature', window=24, n_windows=8)
    mvt_summary.to_csv('outputs/figures/mvt_shift_summary.csv', index=False)
    print("Saved mvt shift summary to outputs/figures/mvt_shift_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='synth', choices=['synth','data'], help='synth -> tạo dữ liệu giả, data -> dùng file thật')
    parser.add_argument('--data', default=None, help='đường dẫn file CSV (nếu mode=data)')
    parser.add_argument('--hours', type=int, default=24*30, help='số giờ nếu mode=synth')
    parser.add_argument('--start', default='2025-09-01', help='ngày bắt đầu nếu mode=synth')
    args = parser.parse_args()
    main(args)
