import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Phân tích dữ liệu thời tiết MVT", layout="wide")
st.title("PHÂN TÍCH DỮ LIỆU THỜI TIẾT ỨNG DỤNG ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)")
st.markdown("---")

# ============================================================
# HÀM LẤY DỮ LIỆU
# ============================================================

def get_weather_data(city="Hanoi", api_key=None):
    """Lấy dữ liệu thời tiết 5 ngày (mỗi 3 giờ) từ OpenWeatherMap"""
    if not api_key:
        st.error("Thiếu API key. Hãy nhập vào ô bên dưới hoặc thêm vào phần secrets.")
        return None

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={api_key}&lang=vi"
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error(f"Lỗi truy xuất API: {resp.status_code} — {resp.text}")
        return None

    data = resp.json()
    records = []
    for item in data["list"]:
        records.append({
            "Thời điểm": item["dt_txt"],
            "Nhiệt độ": item["main"]["temp"],
            "Độ ẩm": item["main"]["humidity"],
            "Lượng mưa": item.get("rain", {}).get("3h", 0)
        })
    df = pd.DataFrame(records)
    df["Thời điểm"] = pd.to_datetime(df["Thời điểm"])
    df = df.sort_values("Thời điểm").reset_index(drop=True)
    df["timestamp_days"] = (df["Thời điểm"] - df["Thời điểm"].iloc[0]).dt.total_seconds() / 86400.0
    return df

# ============================================================
# 1. NHẬP DỮ LIỆU
# ============================================================

st.sidebar.header("NHẬP DỮ LIỆU VÀO ỨNG DỤNG")
option = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["Dữ liệu mẫu", "Tải file CSV", "Lấy dữ liệu trực tiếp từ API"],
    index=0
)

if option == "Dữ liệu mẫu":
    df = pd.read_csv("data/sample_weather.csv")
    df.rename(columns={
        "timestamp": "Thời điểm",
        "temperature": "Nhiệt độ",
        "humidity": "Độ ẩm",
        "precip": "Lượng mưa"
    }, inplace=True)
    df["Thời điểm"] = pd.to_datetime(df["Thời điểm"])
    df["timestamp_days"] = (df["Thời điểm"] - df["Thời điểm"].iloc[0]).dt.total_seconds() / 86400.0

elif option == "Tải file CSV":
    uploaded = st.sidebar.file_uploader("Tải lên file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df.rename(columns={
            "timestamp": "Thời điểm",
            "temperature": "Nhiệt độ",
            "humidity": "Độ ẩm",
            "precip": "Lượng mưa"
        }, inplace=True)
        df["Thời điểm"] = pd.to_datetime(df["Thời điểm"])
        df["timestamp_days"] = (df["Thời điểm"] - df["Thời điểm"].iloc[0]).dt.total_seconds() / 86400.0
    else:
        st.warning(" Vui lòng tải file CSV hoặc chọn dữ liệu khác.")
        st.stop()

else:
    city = st.sidebar.text_input(" Nhập tên thành phố:", "Hanoi")
    api_key = st.sidebar.text_input(" Nhập API key OpenWeatherMap:", type="password")
    if st.sidebar.button("Lấy dữ liệu"):
        df = get_weather_data(city, api_key)
        if df is not None:
            st.success(f" Đã tải thành công dữ liệu thời tiết của **{city}**!")
        else:
            st.stop()
    else:
        st.info("Nhập tên thành phố và API key, sau đó nhấn **Lấy dữ liệu**.")
        st.stop()

st.subheader(" 1. Dữ liệu đầu vào")
st.dataframe(df.head())

# ============================================================
# 2. BIỂU ĐỒ MINH HỌA
# ============================================================

st.subheader(" 2. Biểu đồ diễn biến các yếu tố thời tiết")
col = st.selectbox("Chọn yếu tố để phân tích:", ["Nhiệt độ", "Độ ẩm", "Lượng mưa"], index=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Thời điểm"], df[col], marker="o", label=col)
ax.set_xlabel("Thời gian")
ax.set_ylabel(col)
ax.legend()
st.pyplot(fig)

# ============================================================
# 3. PHÂN TÍCH TỐC ĐỘ THAY ĐỔI 
# ============================================================

st.subheader("3. Phân tích tốc độ thay đổi")

df["dt_hours"] = df["Thời điểm"].diff().dt.total_seconds() / 3600.0
df["Chênh lệch"] = df[col].diff()
df["Độ chênh lệch/giờ"] = df["Chênh lệch"] / df["dt_hours"]

avg_per_hour = df["Độ chênh lệch/giờ"].dropna().mean()
std_per_hour = df["Độ chênh lệch/giờ"].dropna().std()
mae_per_hour = np.abs(df["Độ chênh lệch/giờ"].dropna()).mean()

avg_per_day = avg_per_hour * 24

st.markdown("**Bảng giá trị và tốc độ thay đổi từng khoảng:**")
st.dataframe(df[["Thời điểm", col, "Chênh lệch", "Độ chênh lệch/giờ"]].head(10))

st.markdown("**Thống kê tóm tắt:**")
st.markdown(f"- Trung bình (°C/giờ): **{avg_per_hour:.6f}**")
st.markdown(f"- Độ lệch chuẩn (°C/giờ): **{std_per_hour:.4f}**")
st.markdown(f"- Sai số tuyệt đối trung bình (|Δ|): **{mae_per_hour:.4f}**")
st.markdown(f"- Tương đương (°C/ngày): **{avg_per_day:.6f}**")

if np.isnan(avg_per_hour):
    st.warning("Không đủ dữ liệu để tính tốc độ thay đổi.")
else:
    if abs(avg_per_hour) < 1e-6:
        trend = "gần như ổn định"
    elif avg_per_hour > 0:
        trend = f"tăng trung bình {avg_per_hour:.4f} °C mỗi giờ (~{avg_per_day:.6f} °C/ngày)"
    else:
        trend = f"giảm trung bình {abs(avg_per_hour):.4f} °C mỗi giờ (~{abs(avg_per_day):.6f} °C/ngày)"
    st.success(f"→ Nhìn chung, {col} có xu hướng **{trend}** trong giai đoạn quan sát.")

# Giải thích 
with st.expander("Giải thích chi tiết"):
    st.markdown("""
    **Ý nghĩa:**  
    Tốc độ thay đổi trung bình được tính theo công thức:
    """)
    st.latex(r"v_{tb} = \frac{f(t_{i+1}) - f(t_i)}{t_{i+1} - t_i}")
    st.markdown("""Trong đó:""")
    st.latex(r"f(t_i):\ \text{giá trị của đại lượng (ví dụ: nhiệt độ) tại thời điểm } t_i")
    st.latex(r"f(t_{i+1}):\ \text{giá trị của đại lượng tại thời điểm kế tiếp } t_{i+1}")
    st.markdown("""Biểu thức cho biết **độ biến thiên trung bình của đại lượng trên mỗi đơn vị thời gian.**""")

    st.markdown("""**Cách hiểu:**  
    - Nếu kết quả > 0 → đại lượng **tăng** theo thời gian.  
    - Nếu kết quả < 0 → đại lượng **giảm**.  
    - Nếu kết quả = 0 → đại lượng **gần như ổn định**.""")

    st.markdown("""**Ví dụ:**  
    Nếu tốc độ trung bình là -0.20 °C/giờ, nghĩa là cứ mỗi giờ nhiệt độ giảm trung bình 0.20 °C  
    → tương đương giảm khoảng 4.8 °C mỗi ngày.
    """)
    
# ============================================================
# CÁC HÀM PHỤ 
# ============================================================

def compute_derivative_series(df, col):
    t_days = df["timestamp_days"]
    y = df[col]
    n = len(df)
    deriv = np.zeros(n)
    for i in range(n):
        if 0 < i < n - 1:
            dt = t_days.iloc[i + 1] - t_days.iloc[i - 1]
            dy = y.iloc[i + 1] - y.iloc[i - 1]
        elif i == 0:
            dt = t_days.iloc[1] - t_days.iloc[0]
            dy = y.iloc[1] - y.iloc[0]
        else:
            dt = t_days.iloc[-1] - t_days.iloc[-2]
            dy = y.iloc[-1] - y.iloc[-2]
        deriv[i] = dy / dt if dt != 0 else np.nan
    return pd.Series(deriv, name=f"d{col}/dt")

def count_mvt_intervals(df_local, col_name, deriv_series):
    t0 = df_local['timestamp'].iloc[0]
    t_days = (df_local['timestamp'] - t0).dt.total_seconds() / 86400.0
    y = df_local[col_name].to_numpy(dtype=float)
    n = len(y)
    count = 0
    for i in range(n - 1):
        if (t_days[i + 1] - t_days[i]) == 0:
            continue
        S = (y[i + 1] - y[i]) / (t_days[i + 1] - t_days[i])
        d_i = deriv_series.iloc[i]
        d_ip1 = deriv_series.iloc[i + 1]
        if np.isnan(S) or np.isnan(d_i) or np.isnan(d_ip1):
            continue
        if (d_i - S) * (d_ip1 - S) < 0 or (d_i - S) == 0 or (d_ip1 - S) == 0:
            count += 1
    return count

# ============================================================
# 4. ẢNH HƯỞNG CỦA LÀM TRÒN DỮ LIỆU
# ============================================================

st.subheader(" 4. Ảnh hưởng của việc làm tròn dữ liệu đến đạo hàm & giá trị trung bình")

rounding_levels = [-1, 0, 1, 2, 3]  # -1 = làm tròn đến hàng chục
deriv_orig = compute_derivative_series(df, col)
summary_rows = []

for k in rounding_levels:
    df_r = df.copy()
    df_r[col] = df_r[col].round(k)
    deriv_r = compute_derivative_series(df_r, col)

    mask = (~deriv_orig.isna()) & (~deriv_r.isna())
    if mask.sum() > 0:
        mae = float(np.nanmean(np.abs(deriv_orig[mask] - deriv_r[mask])))
        max_err = float(np.nanmax(np.abs(deriv_orig[mask] - deriv_r[mask])))
        sign_changes = int(np.sum((np.sign(deriv_orig[mask]) * np.sign(deriv_r[mask])) < 0))
        sign_change_pct = float(sign_changes / mask.sum() * 100.0)
    else:
        mae, max_err, sign_changes, sign_change_pct = np.nan, np.nan, 0, np.nan

    mvt_count_orig = count_mvt_intervals(df, col, deriv_orig)
    mvt_count_round = count_mvt_intervals(df_r, col, deriv_r)

    summary_rows.append({
        "Mức làm tròn": (
            "Đến hàng chục" if k == -1 else f"{k} chữ số sau dấu phẩy"
        ),
        "Sai số trung bình đạo hàm": mae,
        "Sai số lớn nhất đạo hàm": max_err,
        "% thay đổi dấu đạo hàm": round(sign_change_pct, 2),
        "Số khoảng MVT (gốc)": mvt_count_orig,
        "Số khoảng MVT (sau làm tròn)": mvt_count_round
    })


summary_df = pd.DataFrame(summary_rows)
st.markdown(" Bảng so sánh ảnh hưởng của các mức làm tròn:")
st.dataframe(summary_df)

#  SLIDER 
st.markdown("---")
st.subheader(" 5. Minh họa trực quan mức độ làm tròn dữ liệu")

round_options = {
    "Đến hàng chục": -1,
    "Đến hàng đơn vị": 0,
    "1 chữ số sau dấu phẩy": 1,
    "2 chữ số sau dấu phẩy": 2,
    "3 chữ số sau dấu phẩy": 3
}

round_label = st.selectbox("Chọn mức làm tròn dữ liệu:", list(round_options.keys()), index=2)
round_level = round_options[round_label]

df_rounded = df.copy()
df_rounded[col] = df_rounded[col].round(round_level)


show_derivative = st.checkbox("Hiển thị đạo hàm (tốc độ thay đổi) trên biểu đồ", value=False)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["timestamp"], df[col], label="Dữ liệu gốc", alpha=0.7)
ax2.plot(df_rounded["timestamp"], df_rounded[col], "--", label=f"Làm tròn {round_level} chữ số", color="orange")

if show_derivative:
    deriv_orig_plot = compute_derivative_series(df, col)
    deriv_round_plot = compute_derivative_series(df_rounded, col)
    ax2.plot(df["timestamp"], deriv_orig_plot, "g-", alpha=0.5, label="Đạo hàm (gốc)")
    ax2.plot(df["timestamp"], deriv_round_plot, "r--", alpha=0.5, label="Đạo hàm (làm tròn)")

ax2.set_title("Ảnh hưởng của việc làm tròn dữ liệu đến biến thiên và đạo hàm")
ax2.set_xlabel("Thời gian")
ax2.set_ylabel("Nhiệt độ")
ax2.legend()
st.pyplot(fig2)

st.caption("""
Kéo thanh trượt để quan sát mức độ làm tròn ảnh hưởng đến biến thiên và tốc độ thay đổi của dữ liệu.
Làm tròn quá lớn khiến dữ liệu mất chi tiết, đạo hàm bị dao động mạnh và sai lệch trong phân tích hoặc dự báo.
""")

# ============================================================
#  6. MINH HỌA CÔNG THỨC XẤP XỈ TUYẾN TÍNH
# ============================================================

st.markdown("---")
st.subheader("6. Minh họa công thức xấp xỉ tuyến tính f(b) ≈ f(a) + f'(a)(x-a)")

max_i = max(0, len(df) - 2)
idx = st.number_input("Chọn chỉ số i để minh họa (xét khoảng i → i+1):", min_value=0, max_value=max_i, value=0, step=1)
a_idx = int(idx)
b_idx = a_idx + 1

f_a = float(df[col].iloc[a_idx])
f_b = float(df[col].iloc[b_idx])
fprime_a = deriv_orig.iloc[a_idx]
t0 = df["timestamp"].iloc[0]
t_days = (df["timestamp"] - t0).dt.total_seconds() / 86400.0
dt_ab = t_days.iloc[b_idx] - t_days.iloc[a_idx]

if np.isnan(fprime_a):
    st.warning(" Không có giá trị đạo hàm tại điểm a để minh họa.")
else:
    f_approx = f_a + fprime_a * dt_ab
    err = f_b - f_approx
    pct_err = (err / f_b * 100.0) if f_b != 0 else np.nan

    x_lin = np.linspace(t_days.iloc[a_idx] - dt_ab * 0.2, t_days.iloc[b_idx] + dt_ab * 0.2, 100)
    y_lin = f_a + fprime_a * (x_lin - t_days.iloc[a_idx])
    x_lin_ts = t0 + pd.to_timedelta(x_lin, unit="D")

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(df["timestamp"], df[col], marker="o", label="Dữ liệu gốc")
    ax3.scatter([df["timestamp"].iloc[a_idx]], [f_a], color="green", s=80, label="a (tuyến tính hóa)")
    ax3.scatter([df["timestamp"].iloc[b_idx]], [f_b], color="red", s=80, label="b (thực)")
    ax3.plot(x_lin_ts, y_lin, linestyle="--", color="orange", label="Đường xấp xỉ f(a)+f'(a)(x-a)")
    ax3.legend()
    st.pyplot(fig3)

    st.markdown(f"- **Giá trị f(b)** = {f_b:.6g}")
    st.markdown(f"- **Giá trị xấp xỉ f(a)+f'(a)(x-a)** = {f_approx:.6g}")
    st.markdown(f"- **Sai số tuyệt đối:** {err:.6g}")
    st.markdown(f"- **Sai số tương đối:** {pct_err:.3f}%")

st.markdown("---")
st.markdown("###  Kết luận:")
st.markdown("""
- Làm tròn dữ liệu khiến chi tiết nhỏ bị mất và có thể đảo dấu đạo hàm.
- Làm tròn quá mức khiến đạo hàm và điểm MVT bị sai lệch.
- Sai số trung bình và cực đại tăng theo mức làm tròn.
- Nếu sử dụng đạo hàm để dự báo ngắn hạn, việc làm tròn thô có thể dẫn đến sai hướng.
- Hạn chế làm tròn trước khi tính đạo hàm; nếu cần, nên dùng phương pháp làm mịn (smoothing) thay vì làm tròn cứng.
""")
