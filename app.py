import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ================================
# 🌤 WEATHER MVT ANALYZER
# ================================

st.set_page_config(page_title="Weather MVT Analyzer", layout="wide")

st.title("🌤 ỨNG DỤNG PHÂN TÍCH DỮ LIỆU THỜI TIẾT THEO ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)")
st.markdown("---")

# ============================================================
# 1️⃣ HÀM TIỆN ÍCH
# ============================================================

def get_weather_data(city="Hanoi", api_key=None):
    """Lấy dữ liệu thời tiết 5 ngày (mỗi 3 giờ) từ OpenWeatherMap"""
    if not api_key:
        st.error("⛔ Thiếu API key. Hãy thêm vào secrets hoặc nhập trực tiếp.")
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
            "timestamp": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "precip": item.get("rain", {}).get("3h", 0)
        })
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# 2️⃣ NHẬP DỮ LIỆU
# ============================================================

st.sidebar.header("🗂 NHẬP DỮ LIỆU")

option = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["Dữ liệu mẫu", "Tải file CSV", "Lấy trực tiếp từ API"],
    index=0
)

if option == "Dữ liệu mẫu":
    df = pd.read_csv("data/sample_weather.csv")

elif option == "Tải file CSV":
    uploaded = st.sidebar.file_uploader("Tải file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning("⛔ Vui lòng tải file CSV hoặc chọn dữ liệu khác.")
        st.stop()

else:
    city = st.sidebar.text_input("Nhập tên thành phố:", "Hanoi")
    api_key = st.sidebar.text_input("🔑 Nhập API key OpenWeatherMap:", type="password")
    if st.sidebar.button("📡 Lấy dữ liệu"):
        df = get_weather_data(city, api_key)
        if df is not None:
            st.success(f"✅ Đã tải dữ liệu thời tiết của **{city}** thành công!")
        else:
            st.stop()
    else:
        st.info("Nhập thành phố và API key rồi bấm **Lấy dữ liệu**.")
        st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
st.subheader("📊 Dữ liệu đầu vào")
st.dataframe(df.head())

# ============================================================
# 3️⃣ BIỂU ĐỒ MINH HỌA
# ============================================================

st.subheader("📈 Biểu đồ minh họa dữ liệu thời tiết")
col = st.selectbox("Chọn yếu tố để phân tích:", ["temperature", "humidity", "precip"], index=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["timestamp"], df[col], marker="o", label=col)
ax.set_xlabel("Thời gian")
ax.set_ylabel(col)
ax.legend()
st.pyplot(fig)

# ============================================================
# 4️⃣ TÍNH TOÁN CƠ BẢN
# ============================================================

st.subheader("🧮 Tính toán tốc độ thay đổi")

df["slope"] = df[col].diff()
df["derivative"] = (df[col].shift(-1) - df[col].shift(1)) / 2

avg_slope = np.mean(df["slope"].dropna())
st.success(f"📈 Tốc độ thay đổi trung bình của {col} ≈ {avg_slope:.3f}")

# ============================================================
# 5️⃣ PHÂN TÍCH THEO ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)
# ============================================================

st.subheader("📍 Phân tích theo Định lý Giá trị Trung bình (MVT)")

results = []
for i in range(1, len(df) - 1):
    x1, x2 = df["timestamp"].iloc[i - 1], df["timestamp"].iloc[i]
    f1, f2 = df[col].iloc[i - 1], df[col].iloc[i]
    avg_rate = (f2 - f1)
    c = (f2 - f1) / 2 + f1
    results.append({
        "Khoảng": f"[{x1.strftime('%H:%M')}, {x2.strftime('%H:%M')}]",
        "Δf": round(f2 - f1, 3),
        "Tốc độ TB": round(avg_rate, 3),
        "Điểm MVT (xấp xỉ)": round(c, 3)
    })

mvt_df = pd.DataFrame(results)
st.dataframe(mvt_df.head(10))

# ============================================================
# 6️⃣ GIẢI THÍCH & PHÂN TÍCH LÀM TRÒN DỮ LIỆU
# ============================================================

st.subheader("🧠 Giải thích & Phân tích làm tròn dữ liệu (0–3 chữ số)")

# Chuẩn bị dữ liệu thời gian (tính theo ngày)
t0 = df["timestamp"].iloc[0]
df["timestamp_days"] = (df["timestamp"] - t0).dt.total_seconds() / 86400.0

# ---- Hàm tính đạo hàm xấp xỉ ----
def compute_derivative_series(df_local, col_name):
    t_days = df_local["timestamp_days"]
    y = df_local[col_name]
    n = len(df_local)
    d = np.zeros(n)
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
        d[i] = dy / dt if dt != 0 else np.nan
    return pd.Series(d, name=f"d{col_name}/dt")

# ---- Hàm đếm số khoảng MVT ----
def count_mvt_intervals(df_local, col_name, deriv_series):
    t0 = df_local["timestamp"].iloc[0]
    t_days = (df_local["timestamp"] - t0).dt.total_seconds() / 86400.0
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

# ---- So sánh 4 mức làm tròn ----
rounding_levels = [0, 1, 2, 3]
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
        mae = np.nan
        max_err = np.nan
        sign_changes = 0
        sign_change_pct = np.nan

    mvt_count_orig = count_mvt_intervals(df, col, deriv_orig)
    mvt_count_round = count_mvt_intervals(df_r, col, deriv_r)

    summary_rows.append({
        "Làm tròn (chữ số)": k,
        "MAE đạo hàm": mae,
        "Max error": max_err,
        "% đổi dấu đạo hàm": round(sign_change_pct, 2) if not np.isnan(sign_change_pct) else np.nan,
        "Số khoảng MVT (gốc)": mvt_count_orig,
        "Số khoảng MVT (làm tròn)": mvt_count_round
    })

summary_df = pd.DataFrame(summary_rows)
st.markdown("**📋 Bảng so sánh tóm tắt ảnh hưởng của làm tròn**")
st.dataframe(summary_df)

# ---- Chi tiết từng mức làm tròn ----
for k in rounding_levels:
    with st.expander(f"Chi tiết: làm tròn {k} chữ số sau dấu phẩy"):
        df_r = df.copy()
        df_r[col] = df_r[col].round(k)
        deriv_r = compute_derivative_series(df_r, col)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(df["timestamp"], df[col], marker="o", label="Gốc", alpha=0.8)
        axes[0].plot(df_r["timestamp"], df_r[col], marker="x", linestyle="--", label=f"Làm tròn {k}", alpha=0.9)
        axes[0].set_title(f"Dữ liệu gốc vs làm tròn ({k} chữ số)")
        axes[0].legend()

        axes[1].plot(df["timestamp"], deriv_orig, marker="o", label="Đạo hàm gốc")
        axes[1].plot(df_r["timestamp"], deriv_r, marker="x", linestyle="--", label=f"Đạo hàm làm tròn {k}")
        axes[1].set_title("Đạo hàm xấp xỉ (value/day)")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# 🔎 Minh họa công thức xấp xỉ tuyến tính
# ============================================================

st.markdown("---")
st.subheader("🔎 Minh họa công thức xấp xỉ tuyến tính f(b) ≈ f(a) + f'(a)(x-a)")

max_i = max(0, len(df) - 2)
idx = st.number_input("Chọn chỉ số i để minh họa (xét khoảng i → i+1)", min_value=0, max_value=max_i, value=0, step=1)
a_idx = int(idx)
b_idx = a_idx + 1

f_a = float(df[col].iloc[a_idx])
f_b = float(df[col].iloc[b_idx])
fprime_a = deriv_orig.iloc[a_idx]
t0 = df["timestamp"].iloc[0]
t_days = (df["timestamp"] - t0).dt.total_seconds() / 86400.0
dt_ab = t_days.iloc[b_idx] - t_days.iloc[a_idx]

if np.isnan(fprime_a):
    st.warning("Không có giá trị đạo hàm tại điểm a để minh họa.")
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

    st.markdown(f"- **f(b)** = {f_b:.6g}")
    st.markdown(f"- **f_approx** = {f_approx:.6g}")
    st.markdown(f"- **Sai số tuyệt đối:** {err:.6g}")
    st.markdown(f"- **Sai số tương đối:** {pct_err:.3f}%")

# ============================================================
# ✅ KẾT LUẬN
# ============================================================

st.markdown("---")
st.markdown("### ✅ Kết luận:")
st.markdown("""
- Làm tròn làm mất chi tiết nhỏ và có thể đảo dấu đạo hàm.
- Làm tròn quá mức khiến đạo hàm và điểm MVT bị sai lệch.
- MAE/Max Error đạo hàm tăng theo mức làm tròn.
- Nếu dùng đạo hàm để dự báo ngắn hạn, việc làm tròn thô có thể gây dự báo sai hướng.
- 👉 Hạn chế làm tròn trước khi tính đạo hàm; nếu cần, nên làm mịn (smoothing) thay vì làm tròn thô.
""")
