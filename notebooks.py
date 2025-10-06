# Experiment: Rounding Effects on Weather MVT Analyzer
#Mục tiêu: áp dụng pipeline giống repo gốc nhưng dùng dữ liệu thời tiết; khảo sát ảnh hưởng của làm tròn.

# Imports
import pandas as pd
from src.preprocess import load_weather_data, apply_rounding_multi, ensure_time_index
from src.generate_synthetic_weather import generate_synthetic_weather
from src.mvt_core import find_mvt_points, mvt_shift_summary
from src.model_eval import evaluate
from src.visualize import plot_temp_rounded, plot_error_vs_rounding

# Tạo dữ liệu giả (hoặc load data thật)
df = generate_synthetic_weather(hours=24*30)
df.to_csv('data/raw_weather.csv', index=False)
df = load_weather_data('data/raw_weather.csv')

# Áp rounding và evaluate
rounding_schemes = {'temperature': {'type':'decimals','levels':[2,1,0,-1]}, 'humidity': {'type':'step','levels':[1,5,10]}}
variants = apply_rounding_multi(df, rounding_schemes)
results = []
for v in variants:
    metrics, model, _ = evaluate(v['df'], feature='temperature')
    results.append({'feature':v['feature'],'level':v['level'], **metrics})
results_df = pd.DataFrame(results)
results_df

# Vẽ một số biểu đồ
plot_error_vs_rounding(results_df, feature='temperature')
