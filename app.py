import sys, os
sys.path.append(os.path.abspath("src"))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Cấu hình đường dẫn để Python luôn thấy thư mục src/
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ✅ Import đúng theo cấu trúc gốc
from src.generate_synthetic_weather import generate_synthetic_weather
from src.preprocess import load_weather_data, apply_rounding_single, ensure_time_index
from src.model_eval import evaluate
from src.mvt_core import find_mvt_points, mvt_shift_summary
from src.visualize import plot_temp_rounded, plot_error_vs_rounding, plot_mvt_points

st.set_page_config(page_title="MVT Weather Analyzer", layout="wide")

st.title("🌦️ MVT Weather Analyzer")
st.write("Phân tích dữ liệu thời tiết bằng định lý giá trị trung bình (MVT) và khảo sát ảnh hưởng của việc làm tròn dữ liệu.")

# --- Chọn nguồn dữ liệu ---
mode = st.radio("Chọn nguồn dữ liệu:", ["Dữ liệu giả (synthetic)", "Tải lên file CSV"])

if mode == "Dữ liệu giả (synthetic)":
    hours = st.slider("Số giờ dữ liệu:", 24, 24*90, 24*30, step=24)
    df = generate_synthetic_weather(hours=hours)
else:
    uploaded = st.file_uploader("Tải lên file CSV thời tiết:", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=["time"])
    else:
        st.stop()

df = ensure_time_index(df)
st.subheader("📋 Mẫu dữ liệu ban đầu:")
st.dataframe(df.head())

# --- Cấu hình làm tròn ---
col = st.selectbox("Chọn cột để làm tròn:", df.columns.drop("time"))
rtype = st.radio("Kiểu làm tròn:", ["decimals", "step"])

if rtype == "decimals":
    levels = st.multiselect("Số chữ số thập phân (decimals):", [2, 1, 0, -1], default=[2, 1, 0])
else:
    levels = st.multiselect("Step (bước làm tròn):", [0.1, 0.5, 1, 5, 10], default=[1, 5])

if st.button("🚀 Chạy phân tích"):
    all_results = []
    fig_list = []
    for lvl in levels:
        # áp dụng rounding
        scheme = {"type": rtype, "levels": [lvl]}
        df_r = apply_rounding_single(df, col, scheme)[0][1]

        # vẽ so sánh
        st.markdown(f"### 🔍 So sánh cho {col}, mức làm tròn = `{lvl}`")
        plot_temp_rounded(df, df_r, days=5)

        # evaluate
        metrics, model, _ = evaluate(df_r, feature='temperature')
        all_results.append({
            "feature": col,
            "level": lvl,
            "MSE": metrics["mse"],
            "MAE": metrics["mae"],
            "R2": metrics["r2"]
        })

        # MVT points
        mvt_idx = find_mvt_points(df_r["temperature"].values, window=24)
        plot_mvt_points(df_r, mvt_idx, feature='temperature')

    res_df = pd.DataFrame(all_results)
    st.subheader("📊 Kết quả đánh giá")
    st.dataframe(res_df)

    # vẽ tổng hợp ảnh hưởng rounding
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(res_df["level"], res_df["MSE"], marker="o")
    ax.set_xlabel("Mức làm tròn")
    ax.set_ylabel("MSE")
    ax.set_title("Ảnh hưởng của làm tròn tới MSE (temperature)")
    st.pyplot(fig)
